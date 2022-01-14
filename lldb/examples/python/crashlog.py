#!/usr/bin/env python3

#----------------------------------------------------------------------
# Be sure to add the python path that points to the LLDB shared library.
#
# To use this in the embedded python interpreter using "lldb":
#
#   cd /path/containing/crashlog.py
#   lldb
#   (lldb) script import crashlog
#   "crashlog" command installed, type "crashlog --help" for detailed help
#   (lldb) crashlog ~/Library/Logs/DiagnosticReports/a.crash
#
# The benefit of running the crashlog command inside lldb in the
# embedded python interpreter is when the command completes, there
# will be a target with all of the files loaded at the locations
# described in the crash log. Only the files that have stack frames
# in the backtrace will be loaded unless the "--load-all" option
# has been specified. This allows users to explore the program in the
# state it was in right at crash time.
#
# On MacOSX csh, tcsh:
#   ( setenv PYTHONPATH /path/to/LLDB.framework/Resources/Python ; ./crashlog.py ~/Library/Logs/DiagnosticReports/a.crash )
#
# On MacOSX sh, bash:
#   PYTHONPATH=/path/to/LLDB.framework/Resources/Python ./crashlog.py ~/Library/Logs/DiagnosticReports/a.crash
#----------------------------------------------------------------------

from __future__ import print_function
import cmd
import datetime
import glob
import optparse
import os
import platform
import plistlib
import re
import shlex
import string
import subprocess
import sys
import time
import uuid
import json

try:
    # First try for LLDB in case PYTHONPATH is already correctly setup.
    import lldb
except ImportError:
    # Ask the command line driver for the path to the lldb module. Copy over
    # the environment so that SDKROOT is propagated to xcrun.
    env = os.environ.copy()
    env['LLDB_DEFAULT_PYTHON_VERSION'] = str(sys.version_info.major)
    command =  ['xcrun', 'lldb', '-P'] if platform.system() == 'Darwin' else ['lldb', '-P']
    # Extend the PYTHONPATH if the path exists and isn't already there.
    lldb_python_path = subprocess.check_output(command, env=env).decode("utf-8").strip()
    if os.path.exists(lldb_python_path) and not sys.path.__contains__(lldb_python_path):
        sys.path.append(lldb_python_path)
    # Try importing LLDB again.
    try:
        import lldb
    except ImportError:
        print("error: couldn't locate the 'lldb' module, please set PYTHONPATH correctly")
        sys.exit(1)

from lldb.utils import symbolication


def read_plist(s):
    if sys.version_info.major == 3:
        return plistlib.loads(s)
    else:
        return plistlib.readPlistFromString(s)

class CrashLog(symbolication.Symbolicator):
    class Thread:
        """Class that represents a thread in a darwin crash log"""

        def __init__(self, index, app_specific_backtrace):
            self.index = index
            self.frames = list()
            self.idents = list()
            self.registers = dict()
            self.reason = None
            self.queue = None
            self.crashed = False
            self.app_specific_backtrace = app_specific_backtrace

        def dump(self, prefix):
            if self.app_specific_backtrace:
                print("%Application Specific Backtrace[%u] %s" % (prefix, self.index, self.reason))
            else:
                print("%sThread[%u] %s" % (prefix, self.index, self.reason))
            if self.frames:
                print("%s  Frames:" % (prefix))
                for frame in self.frames:
                    frame.dump(prefix + '    ')
            if self.registers:
                print("%s  Registers:" % (prefix))
                for reg in self.registers.keys():
                    print("%s    %-8s = %#16.16x" % (prefix, reg, self.registers[reg]))

        def dump_symbolicated(self, crash_log, options):
            this_thread_crashed = self.app_specific_backtrace
            if not this_thread_crashed:
                this_thread_crashed = self.did_crash()
                if options.crashed_only and this_thread_crashed == False:
                    return

            print("%s" % self)
            display_frame_idx = -1
            for frame_idx, frame in enumerate(self.frames):
                disassemble = (
                    this_thread_crashed or options.disassemble_all_threads) and frame_idx < options.disassemble_depth
                if frame_idx == 0:
                    symbolicated_frame_addresses = crash_log.symbolicate(
                        frame.pc & crash_log.addr_mask, options.verbose)
                else:
                    # Any frame above frame zero and we have to subtract one to
                    # get the previous line entry
                    symbolicated_frame_addresses = crash_log.symbolicate(
                        (frame.pc & crash_log.addr_mask) - 1, options.verbose)

                if symbolicated_frame_addresses:
                    symbolicated_frame_address_idx = 0
                    for symbolicated_frame_address in symbolicated_frame_addresses:
                        display_frame_idx += 1
                        print('[%3u] %s' % (frame_idx, symbolicated_frame_address))
                        if (options.source_all or self.did_crash(
                        )) and display_frame_idx < options.source_frames and options.source_context:
                            source_context = options.source_context
                            line_entry = symbolicated_frame_address.get_symbol_context().line_entry
                            if line_entry.IsValid():
                                strm = lldb.SBStream()
                                if line_entry:
                                    crash_log.debugger.GetSourceManager().DisplaySourceLinesWithLineNumbers(
                                        line_entry.file, line_entry.line, source_context, source_context, "->", strm)
                                source_text = strm.GetData()
                                if source_text:
                                    # Indent the source a bit
                                    indent_str = '    '
                                    join_str = '\n' + indent_str
                                    print('%s%s' % (indent_str, join_str.join(source_text.split('\n'))))
                        if symbolicated_frame_address_idx == 0:
                            if disassemble:
                                instructions = symbolicated_frame_address.get_instructions()
                                if instructions:
                                    print()
                                    symbolication.disassemble_instructions(
                                        crash_log.get_target(),
                                        instructions,
                                        frame.pc,
                                        options.disassemble_before,
                                        options.disassemble_after,
                                        frame.index > 0)
                                    print()
                        symbolicated_frame_address_idx += 1
                else:
                    print(frame)
            if self.registers:
                print()
                for reg in self.registers.keys():
                    print("    %-8s = %#16.16x" % (reg, self.registers[reg]))
            elif self.crashed:
               print()
               print("No thread state (register information) available")

        def add_ident(self, ident):
            if ident not in self.idents:
                self.idents.append(ident)

        def did_crash(self):
            return self.reason is not None

        def __str__(self):
            if self.app_specific_backtrace:
                s = "Application Specific Backtrace[%u]" % self.index
            else:
                s = "Thread[%u]" % self.index
            if self.reason:
                s += ' %s' % self.reason
            return s

    class Frame:
        """Class that represents a stack frame in a thread in a darwin crash log"""

        def __init__(self, index, pc, description):
            self.pc = pc
            self.description = description
            self.index = index

        def __str__(self):
            if self.description:
                return "[%3u] 0x%16.16x %s" % (
                    self.index, self.pc, self.description)
            else:
                return "[%3u] 0x%16.16x" % (self.index, self.pc)

        def dump(self, prefix):
            print("%s%s" % (prefix, str(self)))

    class DarwinImage(symbolication.Image):
        """Class that represents a binary images in a darwin crash log"""
        dsymForUUIDBinary = '/usr/local/bin/dsymForUUID'
        if not os.path.exists(dsymForUUIDBinary):
            try:
                dsymForUUIDBinary = subprocess.check_output('which dsymForUUID',
                                                            shell=True).decode("utf-8").rstrip('\n')
            except:
                dsymForUUIDBinary = ""

        dwarfdump_uuid_regex = re.compile(
            'UUID: ([-0-9a-fA-F]+) \(([^\(]+)\) .*')

        def __init__(
                self,
                text_addr_lo,
                text_addr_hi,
                identifier,
                version,
                uuid,
                path,
                verbose):
            symbolication.Image.__init__(self, path, uuid)
            self.add_section(
                symbolication.Section(
                    text_addr_lo,
                    text_addr_hi,
                    "__TEXT"))
            self.identifier = identifier
            self.version = version
            self.verbose = verbose

        def show_symbol_progress(self):
            """
            Hide progress output and errors from system frameworks as they are plentiful.
            """
            if self.verbose:
                return True
            return not (self.path.startswith("/System/Library/") or
                        self.path.startswith("/usr/lib/"))


        def find_matching_slice(self):
            dwarfdump_cmd_output = subprocess.check_output(
                'dwarfdump --uuid "%s"' % self.path, shell=True).decode("utf-8")
            self_uuid = self.get_uuid()
            for line in dwarfdump_cmd_output.splitlines():
                match = self.dwarfdump_uuid_regex.search(line)
                if match:
                    dwarf_uuid_str = match.group(1)
                    dwarf_uuid = uuid.UUID(dwarf_uuid_str)
                    if self_uuid == dwarf_uuid:
                        self.resolved_path = self.path
                        self.arch = match.group(2)
                        return True
            if not self.resolved_path:
                self.unavailable = True
                if self.show_symbol_progress():
                    print(("error\n    error: unable to locate '%s' with UUID %s"
                           % (self.path, self.get_normalized_uuid_string())))
                return False

        def locate_module_and_debug_symbols(self):
            # Don't load a module twice...
            if self.resolved:
                return True
            # Mark this as resolved so we don't keep trying
            self.resolved = True
            uuid_str = self.get_normalized_uuid_string()
            if self.show_symbol_progress():
                print('Getting symbols for %s %s...' % (uuid_str, self.path), end=' ')
            if os.path.exists(self.dsymForUUIDBinary):
                dsym_for_uuid_command = '%s %s' % (
                    self.dsymForUUIDBinary, uuid_str)
                s = subprocess.check_output(dsym_for_uuid_command, shell=True)
                if s:
                    try:
                        plist_root = read_plist(s)
                    except:
                        print(("Got exception: ", sys.exc_info()[1], " handling dsymForUUID output: \n", s))
                        raise
                    if plist_root:
                        plist = plist_root[uuid_str]
                        if plist:
                            if 'DBGArchitecture' in plist:
                                self.arch = plist['DBGArchitecture']
                            if 'DBGDSYMPath' in plist:
                                self.symfile = os.path.realpath(
                                    plist['DBGDSYMPath'])
                            if 'DBGSymbolRichExecutable' in plist:
                                self.path = os.path.expanduser(
                                    plist['DBGSymbolRichExecutable'])
                                self.resolved_path = self.path
            if not self.resolved_path and os.path.exists(self.path):
                if not self.find_matching_slice():
                    return False
            if not self.resolved_path and not os.path.exists(self.path):
                try:
                    mdfind_results = subprocess.check_output(
                        ["/usr/bin/mdfind",
                         "com_apple_xcode_dsym_uuids == %s" % uuid_str]).decode("utf-8").splitlines()
                    found_matching_slice = False
                    for dsym in mdfind_results:
                        dwarf_dir = os.path.join(dsym, 'Contents/Resources/DWARF')
                        if not os.path.exists(dwarf_dir):
                            # Not a dSYM bundle, probably an Xcode archive.
                            continue
                        print('falling back to binary inside "%s"' % dsym)
                        self.symfile = dsym
                        for filename in os.listdir(dwarf_dir):
                           self.path = os.path.join(dwarf_dir, filename)
                           if self.find_matching_slice():
                              found_matching_slice = True
                              break
                        if found_matching_slice:
                           break
                except:
                    pass
            if (self.resolved_path and os.path.exists(self.resolved_path)) or (
                    self.path and os.path.exists(self.path)):
                print('ok')
                return True
            else:
                self.unavailable = True
            return False

    def __init__(self, debugger, path, verbose):
        """CrashLog constructor that take a path to a darwin crash log file"""
        symbolication.Symbolicator.__init__(self, debugger)
        self.path = os.path.expanduser(path)
        self.info_lines = list()
        self.system_profile = list()
        self.threads = list()
        self.backtraces = list()  # For application specific backtraces
        self.idents = list()  # A list of the required identifiers for doing all stack backtraces
        self.errors = list()
        self.crashed_thread_idx = -1
        self.version = -1
        self.target = None
        self.verbose = verbose

    def dump(self):
        print("Crash Log File: %s" % (self.path))
        if self.backtraces:
            print("\nApplication Specific Backtraces:")
            for thread in self.backtraces:
                thread.dump('  ')
        print("\nThreads:")
        for thread in self.threads:
            thread.dump('  ')
        print("\nImages:")
        for image in self.images:
            image.dump('  ')

    def find_image_with_identifier(self, identifier):
        for image in self.images:
            if image.identifier == identifier:
                return image
        regex_text = '^.*\.%s$' % (re.escape(identifier))
        regex = re.compile(regex_text)
        for image in self.images:
            if regex.match(image.identifier):
                return image
        return None

    def create_target(self):
        if self.target is None:
            self.target = symbolication.Symbolicator.create_target(self)
            if self.target:
                return self.target
            # We weren't able to open the main executable as, but we can still
            # symbolicate
            print('crashlog.create_target()...2')
            if self.idents:
                for ident in self.idents:
                    image = self.find_image_with_identifier(ident)
                    if image:
                        self.target = image.create_target(self.debugger)
                        if self.target:
                            return self.target  # success
            print('crashlog.create_target()...3')
            for image in self.images:
                self.target = image.create_target(self.debugger)
                if self.target:
                    return self.target  # success
            print('crashlog.create_target()...4')
            print('error: Unable to locate any executables from the crash log.')
            print('       Try loading the executable into lldb before running crashlog')
            print('       and/or make sure the .dSYM bundles can be found by Spotlight.')
        return self.target

    def get_target(self):
        return self.target


class CrashLogFormatException(Exception):
    pass


class CrashLogParseException(Exception):
   pass


class CrashLogParser:
    def parse(self, debugger, path, verbose):
        try:
            return JSONCrashLogParser(debugger, path, verbose).parse()
        except CrashLogFormatException:
            return TextCrashLogParser(debugger, path, verbose).parse()


class JSONCrashLogParser:
    def __init__(self, debugger, path, verbose):
        self.path = os.path.expanduser(path)
        self.verbose = verbose
        self.crashlog = CrashLog(debugger, self.path, self.verbose)

    def parse(self):
        with open(self.path, 'r') as f:
            buffer = f.read()

        # Skip the first line if it contains meta data.
        head, _, tail = buffer.partition('\n')
        try:
            metadata = json.loads(head)
            if 'app_name' in metadata and 'app_version' in metadata:
                buffer = tail
        except ValueError:
            pass

        try:
            self.data = json.loads(buffer)
        except ValueError:
            raise CrashLogFormatException()

        try:
            self.parse_process_info(self.data)
            self.parse_images(self.data['usedImages'])
            self.parse_threads(self.data['threads'])
            self.parse_errors(self.data)
            thread = self.crashlog.threads[self.crashlog.crashed_thread_idx]
            reason = self.parse_crash_reason(self.data['exception'])
            if thread.reason:
                thread.reason = '{} {}'.format(thread.reason, reason)
            else:
                thread.reason = reason
        except (KeyError, ValueError, TypeError) as e:
            raise CrashLogParseException(
                'Failed to parse JSON crashlog: {}: {}'.format(
                    type(e).__name__, e))

        return self.crashlog

    def get_used_image(self, idx):
        return self.data['usedImages'][idx]

    def parse_process_info(self, json_data):
        self.crashlog.process_id = json_data['pid']
        self.crashlog.process_identifier = json_data['procName']
        self.crashlog.process_path = json_data['procPath']

    def parse_crash_reason(self, json_exception):
        exception_type = json_exception['type']
        exception_signal = json_exception['signal']
        if 'codes' in json_exception:
            exception_extra = " ({})".format(json_exception['codes'])
        elif 'subtype' in json_exception:
            exception_extra = " ({})".format(json_exception['subtype'])
        else:
            exception_extra = ""
        return "{} ({}){}".format(exception_type, exception_signal,
                                  exception_extra)

    def parse_images(self, json_images):
        idx = 0
        for json_image in json_images:
            img_uuid = uuid.UUID(json_image['uuid'])
            low = int(json_image['base'])
            high = int(0)
            name = json_image['name'] if 'name' in json_image else ''
            path = json_image['path'] if 'path' in json_image else ''
            version = ''
            darwin_image = self.crashlog.DarwinImage(low, high, name, version,
                                                     img_uuid, path,
                                                     self.verbose)
            self.crashlog.images.append(darwin_image)
            idx += 1

    def parse_frames(self, thread, json_frames):
        idx = 0
        for json_frame in json_frames:
            image_id = int(json_frame['imageIndex'])
            json_image = self.get_used_image(image_id)
            ident = json_image['name'] if 'name' in json_image else ''
            thread.add_ident(ident)
            if ident not in self.crashlog.idents:
                self.crashlog.idents.append(ident)

            frame_offset = int(json_frame['imageOffset'])
            image_addr = self.get_used_image(image_id)['base']
            pc = image_addr + frame_offset
            thread.frames.append(self.crashlog.Frame(idx, pc, frame_offset))
            idx += 1

    def parse_threads(self, json_threads):
        idx = 0
        for json_thread in json_threads:
            thread = self.crashlog.Thread(idx, False)
            if 'name' in json_thread:
                thread.reason = json_thread['name']
            if json_thread.get('triggered', False):
                self.crashlog.crashed_thread_idx = idx
                thread.crashed = True
                if 'threadState' in json_thread:
                    thread.registers = self.parse_thread_registers(
                        json_thread['threadState'])
            thread.queue = json_thread.get('queue')
            self.parse_frames(thread, json_thread.get('frames', []))
            self.crashlog.threads.append(thread)
            idx += 1

    def parse_thread_registers(self, json_thread_state):
        registers = dict()
        for key, state in json_thread_state.items():
            try:
               value = int(state['value'])
               registers[key] = value
            except (TypeError, ValueError):
               pass
        return registers

    def parse_errors(self, json_data):
       if 'reportNotes' in json_data:
          self.crashlog.errors = json_data['reportNotes']


class CrashLogParseMode:
    NORMAL = 0
    THREAD = 1
    IMAGES = 2
    THREGS = 3
    SYSTEM = 4
    INSTRS = 5


class TextCrashLogParser:
    parent_process_regex = re.compile('^Parent Process:\s*(.*)\[(\d+)\]')
    thread_state_regex = re.compile('^Thread ([0-9]+) crashed with')
    thread_instrs_regex = re.compile('^Thread ([0-9]+) instruction stream')
    thread_regex = re.compile('^Thread ([0-9]+)([^:]*):(.*)')
    app_backtrace_regex = re.compile('^Application Specific Backtrace ([0-9]+)([^:]*):(.*)')
    version = r'(\(.+\)|(arm|x86_)[0-9a-z]+)\s+'
    frame_regex = re.compile(r'^([0-9]+)' r'\s'                # id
                             r'+(.+?)'    r'\s+'               # img_name
                             r'(' +version+ r')?'              # img_version
                             r'(0x[0-9a-fA-F]{7}[0-9a-fA-F]+)' # addr
                             r' +(.*)'                         # offs
                            )
    null_frame_regex = re.compile(r'^([0-9]+)\s+\?\?\?\s+(0{7}0+) +(.*)')
    image_regex_uuid = re.compile(r'(0x[0-9a-fA-F]+)'            # img_lo
                                  r'\s+' '-' r'\s+'              #   -
                                  r'(0x[0-9a-fA-F]+)'     r'\s+' # img_hi
                                  r'[+]?(.+?)'            r'\s+' # img_name
                                  r'(' +version+ ')?'            # img_version
                                  r'(<([-0-9a-fA-F]+)>\s+)?'     # img_uuid
                                  r'(/.*)'                       # img_path
                                 )


    def __init__(self, debugger, path, verbose):
        self.path = os.path.expanduser(path)
        self.verbose = verbose
        self.thread = None
        self.app_specific_backtrace = False
        self.crashlog = CrashLog(debugger, self.path, self.verbose)
        self.parse_mode = CrashLogParseMode.NORMAL
        self.parsers = {
            CrashLogParseMode.NORMAL : self.parse_normal,
            CrashLogParseMode.THREAD : self.parse_thread,
            CrashLogParseMode.IMAGES : self.parse_images,
            CrashLogParseMode.THREGS : self.parse_thread_registers,
            CrashLogParseMode.SYSTEM : self.parse_system,
            CrashLogParseMode.INSTRS : self.parse_instructions,
        }

    def parse(self):
        with open(self.path,'r') as f:
            lines = f.read().splitlines()

        for line in lines:
            line_len = len(line)
            if line_len == 0:
                if self.thread:
                    if self.parse_mode == CrashLogParseMode.THREAD:
                        if self.thread.index == self.crashlog.crashed_thread_idx:
                            self.thread.reason = ''
                            if self.crashlog.thread_exception:
                                self.thread.reason += self.crashlog.thread_exception
                            if self.crashlog.thread_exception_data:
                                self.thread.reason += " (%s)" % self.crashlog.thread_exception_data
                        if self.app_specific_backtrace:
                            self.crashlog.backtraces.append(self.thread)
                        else:
                            self.crashlog.threads.append(self.thread)
                    self.thread = None
                else:
                    # only append an extra empty line if the previous line
                    # in the info_lines wasn't empty
                    if len(self.crashlog.info_lines) > 0 and len(self.crashlog.info_lines[-1]):
                        self.crashlog.info_lines.append(line)
                self.parse_mode = CrashLogParseMode.NORMAL
            else:
                self.parsers[self.parse_mode](line)

        return self.crashlog


    def parse_normal(self, line):
        if line.startswith('Process:'):
            (self.crashlog.process_name, pid_with_brackets) = line[
                8:].strip().split(' [')
            self.crashlog.process_id = pid_with_brackets.strip('[]')
        elif line.startswith('Path:'):
            self.crashlog.process_path = line[5:].strip()
        elif line.startswith('Identifier:'):
            self.crashlog.process_identifier = line[11:].strip()
        elif line.startswith('Version:'):
            version_string = line[8:].strip()
            matched_pair = re.search("(.+)\((.+)\)", version_string)
            if matched_pair:
                self.crashlog.process_version = matched_pair.group(1)
                self.crashlog.process_compatability_version = matched_pair.group(
                    2)
            else:
                self.crashlog.process = version_string
                self.crashlog.process_compatability_version = version_string
        elif self.parent_process_regex.search(line):
            parent_process_match = self.parent_process_regex.search(
                line)
            self.crashlog.parent_process_name = parent_process_match.group(1)
            self.crashlog.parent_process_id = parent_process_match.group(2)
        elif line.startswith('Exception Type:'):
            self.crashlog.thread_exception = line[15:].strip()
            return
        elif line.startswith('Exception Codes:'):
            self.crashlog.thread_exception_data = line[16:].strip()
            return
        elif line.startswith('Exception Subtype:'): # iOS
            self.crashlog.thread_exception_data = line[18:].strip()
            return
        elif line.startswith('Crashed Thread:'):
            self.crashlog.crashed_thread_idx = int(line[15:].strip().split()[0])
            return
        elif line.startswith('Triggered by Thread:'): # iOS
            self.crashlog.crashed_thread_idx = int(line[20:].strip().split()[0])
            return
        elif line.startswith('Report Version:'):
            self.crashlog.version = int(line[15:].strip())
            return
        elif line.startswith('System Profile:'):
            self.parse_mode = CrashLogParseMode.SYSTEM
            return
        elif (line.startswith('Interval Since Last Report:') or
                line.startswith('Crashes Since Last Report:') or
                line.startswith('Per-App Interval Since Last Report:') or
                line.startswith('Per-App Crashes Since Last Report:') or
                line.startswith('Sleep/Wake UUID:') or
                line.startswith('Anonymous UUID:')):
            # ignore these
            return
        elif line.startswith('Thread'):
            thread_state_match = self.thread_state_regex.search(line)
            if thread_state_match:
                self.app_specific_backtrace = False
                thread_state_match = self.thread_regex.search(line)
                thread_idx = int(thread_state_match.group(1))
                self.parse_mode = CrashLogParseMode.THREGS
                self.thread = self.crashlog.threads[thread_idx]
                return
            thread_insts_match  = self.thread_instrs_regex.search(line)
            if thread_insts_match:
                self.parse_mode = CrashLogParseMode.INSTRS
                return
            thread_match = self.thread_regex.search(line)
            if thread_match:
                self.app_specific_backtrace = False
                self.parse_mode = CrashLogParseMode.THREAD
                thread_idx = int(thread_match.group(1))
                self.thread = self.crashlog.Thread(thread_idx, False)
                return
            return
        elif line.startswith('Binary Images:'):
            self.parse_mode = CrashLogParseMode.IMAGES
            return
        elif line.startswith('Application Specific Backtrace'):
            app_backtrace_match = self.app_backtrace_regex.search(line)
            if app_backtrace_match:
                self.parse_mode = CrashLogParseMode.THREAD
                self.app_specific_backtrace = True
                idx = int(app_backtrace_match.group(1))
                self.thread = self.crashlog.Thread(idx, True)
        elif line.startswith('Last Exception Backtrace:'): # iOS
            self.parse_mode = CrashLogParseMode.THREAD
            self.app_specific_backtrace = True
            idx = 1
            self.thread = self.crashlog.Thread(idx, True)
        self.crashlog.info_lines.append(line.strip())

    def parse_thread(self, line):
        if line.startswith('Thread'):
            return
        if self.null_frame_regex.search(line):
            print('warning: thread parser ignored null-frame: "%s"' % line)
            return
        frame_match = self.frame_regex.search(line)
        if frame_match:
            (frame_id, frame_img_name, _, frame_img_version, _,
                frame_addr, frame_ofs) = frame_match.groups()
            ident = frame_img_name
            self.thread.add_ident(ident)
            if ident not in self.crashlog.idents:
                self.crashlog.idents.append(ident)
            self.thread.frames.append(self.crashlog.Frame(int(frame_id), int(
                frame_addr, 0), frame_ofs))
        else:
            print('error: frame regex failed for line: "%s"' % line)

    def parse_images(self, line):
        image_match = self.image_regex_uuid.search(line)
        if image_match:
            (img_lo, img_hi, img_name, _, img_version, _,
                _, img_uuid, img_path) = image_match.groups()
            image = self.crashlog.DarwinImage(int(img_lo, 0), int(img_hi, 0),
                                            img_name.strip(),
                                            img_version.strip()
                                            if img_version else "",
                                            uuid.UUID(img_uuid), img_path,
                                            self.verbose)
            self.crashlog.images.append(image)
        else:
            print("error: image regex failed for: %s" % line)


    def parse_thread_registers(self, line):
        stripped_line = line.strip()
        # "r12: 0x00007fff6b5939c8  r13: 0x0000000007000006  r14: 0x0000000000002a03  r15: 0x0000000000000c00"
        reg_values = re.findall(
            '([a-zA-Z0-9]+: 0[Xx][0-9a-fA-F]+) *', stripped_line)
        for reg_value in reg_values:
            (reg, value) = reg_value.split(': ')
            self.thread.registers[reg.strip()] = int(value, 0)

    def parse_system(self, line):
        self.crashlog.system_profile.append(line)

    def parse_instructions(self, line):
        pass


def usage():
    print("Usage: lldb-symbolicate.py [-n name] executable-image")
    sys.exit(0)


class Interactive(cmd.Cmd):
    '''Interactive prompt for analyzing one or more Darwin crash logs, type "help" to see a list of supported commands.'''
    image_option_parser = None

    def __init__(self, crash_logs):
        cmd.Cmd.__init__(self)
        self.use_rawinput = False
        self.intro = 'Interactive crashlogs prompt, type "help" to see a list of supported commands.'
        self.crash_logs = crash_logs
        self.prompt = '% '

    def default(self, line):
        '''Catch all for unknown command, which will exit the interpreter.'''
        print("uknown command: %s" % line)
        return True

    def do_q(self, line):
        '''Quit command'''
        return True

    def do_quit(self, line):
        '''Quit command'''
        return True

    def do_symbolicate(self, line):
        description = '''Symbolicate one or more darwin crash log files by index to provide source file and line information,
        inlined stack frames back to the concrete functions, and disassemble the location of the crash
        for the first frame of the crashed thread.'''
        option_parser = CreateSymbolicateCrashLogOptions(
            'symbolicate', description, False)
        command_args = shlex.split(line)
        try:
            (options, args) = option_parser.parse_args(command_args)
        except:
            return

        if args:
            # We have arguments, they must valid be crash log file indexes
            for idx_str in args:
                idx = int(idx_str)
                if idx < len(self.crash_logs):
                    SymbolicateCrashLog(self.crash_logs[idx], options)
                else:
                    print('error: crash log index %u is out of range' % (idx))
        else:
            # No arguments, symbolicate all crash logs using the options
            # provided
            for idx in range(len(self.crash_logs)):
                SymbolicateCrashLog(self.crash_logs[idx], options)

    def do_list(self, line=None):
        '''Dump a list of all crash logs that are currently loaded.

        USAGE: list'''
        print('%u crash logs are loaded:' % len(self.crash_logs))
        for (crash_log_idx, crash_log) in enumerate(self.crash_logs):
            print('[%u] = %s' % (crash_log_idx, crash_log.path))

    def do_image(self, line):
        '''Dump information about one or more binary images in the crash log given an image basename, or all images if no arguments are provided.'''
        usage = "usage: %prog [options] <PATH> [PATH ...]"
        description = '''Dump information about one or more images in all crash logs. The <PATH> can be a full path, image basename, or partial path. Searches are done in this order.'''
        command_args = shlex.split(line)
        if not self.image_option_parser:
            self.image_option_parser = optparse.OptionParser(
                description=description, prog='image', usage=usage)
            self.image_option_parser.add_option(
                '-a',
                '--all',
                action='store_true',
                help='show all images',
                default=False)
        try:
            (options, args) = self.image_option_parser.parse_args(command_args)
        except:
            return

        if args:
            for image_path in args:
                fullpath_search = image_path[0] == '/'
                for (crash_log_idx, crash_log) in enumerate(self.crash_logs):
                    matches_found = 0
                    for (image_idx, image) in enumerate(crash_log.images):
                        if fullpath_search:
                            if image.get_resolved_path() == image_path:
                                matches_found += 1
                                print('[%u] ' % (crash_log_idx), image)
                        else:
                            image_basename = image.get_resolved_path_basename()
                            if image_basename == image_path:
                                matches_found += 1
                                print('[%u] ' % (crash_log_idx), image)
                    if matches_found == 0:
                        for (image_idx, image) in enumerate(crash_log.images):
                            resolved_image_path = image.get_resolved_path()
                            if resolved_image_path and string.find(
                                    image.get_resolved_path(), image_path) >= 0:
                                print('[%u] ' % (crash_log_idx), image)
        else:
            for crash_log in self.crash_logs:
                for (image_idx, image) in enumerate(crash_log.images):
                    print('[%u] %s' % (image_idx, image))
        return False


def interactive_crashlogs(debugger, options, args):
    crash_log_files = list()
    for arg in args:
        for resolved_path in glob.glob(arg):
            crash_log_files.append(resolved_path)

    crash_logs = list()
    for crash_log_file in crash_log_files:
        try:
            crash_log = CrashLogParser().parse(debugger, crash_log_file, options.verbose)
        except Exception as e:
            print(e)
            continue
        if options.debug:
            crash_log.dump()
        if not crash_log.images:
            print('error: no images in crash log "%s"' % (crash_log))
            continue
        else:
            crash_logs.append(crash_log)

    interpreter = Interactive(crash_logs)
    # List all crash logs that were imported
    interpreter.do_list()
    interpreter.cmdloop()


def save_crashlog(debugger, command, exe_ctx, result, dict):
    usage = "usage: %prog [options] <output-path>"
    description = '''Export the state of current target into a crashlog file'''
    parser = optparse.OptionParser(
        description=description,
        prog='save_crashlog',
        usage=usage)
    parser.add_option(
        '-v',
        '--verbose',
        action='store_true',
        dest='verbose',
        help='display verbose debug info',
        default=False)
    try:
        (options, args) = parser.parse_args(shlex.split(command))
    except:
        result.PutCString("error: invalid options")
        return
    if len(args) != 1:
        result.PutCString(
            "error: invalid arguments, a single output file is the only valid argument")
        return
    out_file = open(args[0], 'w')
    if not out_file:
        result.PutCString(
            "error: failed to open file '%s' for writing...",
            args[0])
        return
    target = exe_ctx.target
    if target:
        identifier = target.executable.basename
        process = exe_ctx.process
        if process:
            pid = process.id
            if pid != lldb.LLDB_INVALID_PROCESS_ID:
                out_file.write(
                    'Process:         %s [%u]\n' %
                    (identifier, pid))
        out_file.write('Path:            %s\n' % (target.executable.fullpath))
        out_file.write('Identifier:      %s\n' % (identifier))
        out_file.write('\nDate/Time:       %s\n' %
                       (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        out_file.write(
            'OS Version:      Mac OS X %s (%s)\n' %
            (platform.mac_ver()[0], subprocess.check_output('sysctl -n kern.osversion', shell=True).decode("utf-8")))
        out_file.write('Report Version:  9\n')
        for thread_idx in range(process.num_threads):
            thread = process.thread[thread_idx]
            out_file.write('\nThread %u:\n' % (thread_idx))
            for (frame_idx, frame) in enumerate(thread.frames):
                frame_pc = frame.pc
                frame_offset = 0
                if frame.function:
                    block = frame.GetFrameBlock()
                    block_range = block.range[frame.addr]
                    if block_range:
                        block_start_addr = block_range[0]
                        frame_offset = frame_pc - block_start_addr.GetLoadAddress(target)
                    else:
                        frame_offset = frame_pc - frame.function.addr.GetLoadAddress(target)
                elif frame.symbol:
                    frame_offset = frame_pc - frame.symbol.addr.GetLoadAddress(target)
                out_file.write(
                    '%-3u %-32s 0x%16.16x %s' %
                    (frame_idx, frame.module.file.basename, frame_pc, frame.name))
                if frame_offset > 0:
                    out_file.write(' + %u' % (frame_offset))
                line_entry = frame.line_entry
                if line_entry:
                    if options.verbose:
                        # This will output the fullpath + line + column
                        out_file.write(' %s' % (line_entry))
                    else:
                        out_file.write(
                            ' %s:%u' %
                            (line_entry.file.basename, line_entry.line))
                        column = line_entry.column
                        if column:
                            out_file.write(':%u' % (column))
                out_file.write('\n')

        out_file.write('\nBinary Images:\n')
        for module in target.modules:
            text_segment = module.section['__TEXT']
            if text_segment:
                text_segment_load_addr = text_segment.GetLoadAddress(target)
                if text_segment_load_addr != lldb.LLDB_INVALID_ADDRESS:
                    text_segment_end_load_addr = text_segment_load_addr + text_segment.size
                    identifier = module.file.basename
                    module_version = '???'
                    module_version_array = module.GetVersion()
                    if module_version_array:
                        module_version = '.'.join(
                            map(str, module_version_array))
                    out_file.write(
                        '    0x%16.16x - 0x%16.16x  %s (%s - ???) <%s> %s\n' %
                        (text_segment_load_addr,
                         text_segment_end_load_addr,
                         identifier,
                         module_version,
                         module.GetUUIDString(),
                         module.file.fullpath))
        out_file.close()
    else:
        result.PutCString("error: invalid target")


class Symbolicate:
    def __init__(self, debugger, internal_dict):
        pass

    def __call__(self, debugger, command, exe_ctx, result):
        try:
            SymbolicateCrashLogs(debugger, shlex.split(command))
        except Exception as e:
            result.PutCString("error: python exception: %s" % e)

    def get_short_help(self):
        return "Symbolicate one or more darwin crash log files."

    def get_long_help(self):
        option_parser = CrashLogOptionParser()
        return option_parser.format_help()


def SymbolicateCrashLog(crash_log, options):
    if options.debug:
        crash_log.dump()
    if not crash_log.images:
        print('error: no images in crash log')
        return

    if options.dump_image_list:
        print("Binary Images:")
        for image in crash_log.images:
            if options.verbose:
                print(image.debug_dump())
            else:
                print(image)

    target = crash_log.create_target()
    if not target:
        return
    exe_module = target.GetModuleAtIndex(0)
    images_to_load = list()
    loaded_images = list()
    if options.load_all_images:
        # --load-all option was specified, load everything up
        for image in crash_log.images:
            images_to_load.append(image)
    else:
        # Only load the images found in stack frames for the crashed threads
        if options.crashed_only:
            for thread in crash_log.threads:
                if thread.did_crash():
                    for ident in thread.idents:
                        images = crash_log.find_images_with_identifier(ident)
                        if images:
                            for image in images:
                                images_to_load.append(image)
                        else:
                            print('error: can\'t find image for identifier "%s"' % ident)
        else:
            for ident in crash_log.idents:
                images = crash_log.find_images_with_identifier(ident)
                if images:
                    for image in images:
                        images_to_load.append(image)
                else:
                    print('error: can\'t find image for identifier "%s"' % ident)

    for image in images_to_load:
        if image not in loaded_images:
            err = image.add_module(target)
            if err:
                print(err)
            else:
                loaded_images.append(image)

    if crash_log.backtraces:
        for thread in crash_log.backtraces:
            thread.dump_symbolicated(crash_log, options)
            print()

    for thread in crash_log.threads:
        thread.dump_symbolicated(crash_log, options)
        print()

    if crash_log.errors:
        print("Errors:")
        for error in crash_log.errors:
            print(error)


def CreateSymbolicateCrashLogOptions(
        command_name,
        description,
        add_interactive_options):
    usage = "usage: %prog [options] <FILE> [FILE ...]"
    option_parser = optparse.OptionParser(
        description=description, prog='crashlog', usage=usage)
    option_parser.add_option(
        '--verbose',
        '-v',
        action='store_true',
        dest='verbose',
        help='display verbose debug info',
        default=False)
    option_parser.add_option(
        '--debug',
        '-g',
        action='store_true',
        dest='debug',
        help='display verbose debug logging',
        default=False)
    option_parser.add_option(
        '--load-all',
        '-a',
        action='store_true',
        dest='load_all_images',
        help='load all executable images, not just the images found in the crashed stack frames',
        default=False)
    option_parser.add_option(
        '--images',
        action='store_true',
        dest='dump_image_list',
        help='show image list',
        default=False)
    option_parser.add_option(
        '--debug-delay',
        type='int',
        dest='debug_delay',
        metavar='NSEC',
        help='pause for NSEC seconds for debugger',
        default=0)
    option_parser.add_option(
        '--crashed-only',
        '-c',
        action='store_true',
        dest='crashed_only',
        help='only symbolicate the crashed thread',
        default=False)
    option_parser.add_option(
        '--disasm-depth',
        '-d',
        type='int',
        dest='disassemble_depth',
        help='set the depth in stack frames that should be disassembled (default is 1)',
        default=1)
    option_parser.add_option(
        '--disasm-all',
        '-D',
        action='store_true',
        dest='disassemble_all_threads',
        help='enabled disassembly of frames on all threads (not just the crashed thread)',
        default=False)
    option_parser.add_option(
        '--disasm-before',
        '-B',
        type='int',
        dest='disassemble_before',
        help='the number of instructions to disassemble before the frame PC',
        default=4)
    option_parser.add_option(
        '--disasm-after',
        '-A',
        type='int',
        dest='disassemble_after',
        help='the number of instructions to disassemble after the frame PC',
        default=4)
    option_parser.add_option(
        '--source-context',
        '-C',
        type='int',
        metavar='NLINES',
        dest='source_context',
        help='show NLINES source lines of source context (default = 4)',
        default=4)
    option_parser.add_option(
        '--source-frames',
        type='int',
        metavar='NFRAMES',
        dest='source_frames',
        help='show source for NFRAMES (default = 4)',
        default=4)
    option_parser.add_option(
        '--source-all',
        action='store_true',
        dest='source_all',
        help='show source for all threads, not just the crashed thread',
        default=False)
    if add_interactive_options:
        option_parser.add_option(
            '-i',
            '--interactive',
            action='store_true',
            help='parse all crash logs and enter interactive mode',
            default=False)
    return option_parser


def CrashLogOptionParser():
    description = '''Symbolicate one or more darwin crash log files to provide source file and line information,
inlined stack frames back to the concrete functions, and disassemble the location of the crash
for the first frame of the crashed thread.
If this script is imported into the LLDB command interpreter, a "crashlog" command will be added to the interpreter
for use at the LLDB command line. After a crash log has been parsed and symbolicated, a target will have been
created that has all of the shared libraries loaded at the load addresses found in the crash log file. This allows
you to explore the program as if it were stopped at the locations described in the crash log and functions can
be disassembled and lookups can be performed using the addresses found in the crash log.'''
    return CreateSymbolicateCrashLogOptions('crashlog', description, True)

def SymbolicateCrashLogs(debugger, command_args):
    option_parser = CrashLogOptionParser()
    try:
        (options, args) = option_parser.parse_args(command_args)
    except:
        return

    if options.debug:
        print('command_args = %s' % command_args)
        print('options', options)
        print('args', args)

    if options.debug_delay > 0:
        print("Waiting %u seconds for debugger to attach..." % options.debug_delay)
        time.sleep(options.debug_delay)
    error = lldb.SBError()

    if args:
        if options.interactive:
            interactive_crashlogs(debugger, options, args)
        else:
            for crash_log_file in args:
                crash_log = CrashLogParser().parse(debugger, crash_log_file, options.verbose)
                SymbolicateCrashLog(crash_log, options)

if __name__ == '__main__':
    # Create a new debugger instance
    debugger = lldb.SBDebugger.Create()
    SymbolicateCrashLogs(debugger, sys.argv[1:])
    lldb.SBDebugger.Destroy(debugger)

def __lldb_init_module(debugger, internal_dict):
    debugger.HandleCommand(
        'command script add -c lldb.macosx.crashlog.Symbolicate crashlog')
    debugger.HandleCommand(
        'command script add -f lldb.macosx.crashlog.save_crashlog save_crashlog')
