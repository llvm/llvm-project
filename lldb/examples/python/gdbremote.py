#!/usr/bin/python

#----------------------------------------------------------------------
# This module will enable GDB remote packet logging when the 
# 'start_gdb_log' command is called with a filename to log to. When the
# 'stop_gdb_log' command is called, it will disable the logging and 
# print out statistics about how long commands took to execute and also
# will primnt ou
# Be sure to add the python path that points to the LLDB shared library.
#
# To use this in the embedded python interpreter using "lldb" just
# import it with the full path using the "command script import" 
# command. This can be done from the LLDB command line:
#   (lldb) command script import /path/to/gdbremote.py
# Or it can be added to your ~/.lldbinit file so this module is always
# available.
#----------------------------------------------------------------------

import commands
import optparse
import os
import re
import shlex
import string
import sys
import tempfile

#----------------------------------------------------------------------
# Global variables
#----------------------------------------------------------------------
g_log_file = ''
g_byte_order = 'little'

class TerminalColors:
    '''Simple terminal colors class'''
    def __init__(self, enabled = True):
        # TODO: discover terminal type from "file" and disable if
        # it can't handle the color codes
        self.enabled = enabled
    
    def reset(self):
        '''Reset all terminal colors and formatting.'''
        if self.enabled:
            return "\x1b[0m";
        return ''
    
    def bold(self, on = True):
        '''Enable or disable bold depending on the "on" paramter.'''
        if self.enabled:
            if on:
                return "\x1b[1m";
            else:
                return "\x1b[22m";
        return ''
    
    def italics(self, on = True):
        '''Enable or disable italics depending on the "on" paramter.'''
        if self.enabled:
            if on:
                return "\x1b[3m";
            else:
                return "\x1b[23m";
        return ''
    
    def underline(self, on = True):
        '''Enable or disable underline depending on the "on" paramter.'''
        if self.enabled:
            if on:
                return "\x1b[4m";
            else:
                return "\x1b[24m";
        return ''
    
    def inverse(self, on = True):
        '''Enable or disable inverse depending on the "on" paramter.'''
        if self.enabled:
            if on:
                return "\x1b[7m";
            else:
                return "\x1b[27m";
        return ''
    
    def strike(self, on = True):
        '''Enable or disable strike through depending on the "on" paramter.'''
        if self.enabled:
            if on:
                return "\x1b[9m";
            else:                
                return "\x1b[29m";
        return ''
                     
    def black(self, fg = True):        
        '''Set the foreground or background color to black. 
        The foreground color will be set if "fg" tests True. The background color will be set if "fg" tests False.'''
        if self.enabled:         
            if fg:               
                return "\x1b[30m";
            else:
                return "\x1b[40m";
        return ''
    
    def red(self, fg = True):          
        '''Set the foreground or background color to red. 
        The foreground color will be set if "fg" tests True. The background color will be set if "fg" tests False.'''
        if self.enabled:         
            if fg:               
                return "\x1b[31m";
            else:                
                return "\x1b[41m";
        return ''
    
    def green(self, fg = True):        
        '''Set the foreground or background color to green. 
        The foreground color will be set if "fg" tests True. The background color will be set if "fg" tests False.'''
        if self.enabled:         
            if fg:               
                return "\x1b[32m";
            else:                
                return "\x1b[42m";
        return ''
    
    def yellow(self, fg = True):       
        '''Set the foreground or background color to yellow. 
        The foreground color will be set if "fg" tests True. The background color will be set if "fg" tests False.'''
        if self.enabled:         
            if fg:               
                return "\x1b[43m";
            else:                
                return "\x1b[33m";
        return ''
    
    def blue(self, fg = True):         
        '''Set the foreground or background color to blue. 
        The foreground color will be set if "fg" tests True. The background color will be set if "fg" tests False.'''
        if self.enabled:         
            if fg:               
                return "\x1b[34m";
            else:                
                return "\x1b[44m";
        return ''
    
    def magenta(self, fg = True):      
        '''Set the foreground or background color to magenta. 
        The foreground color will be set if "fg" tests True. The background color will be set if "fg" tests False.'''
        if self.enabled:         
            if fg:               
                return "\x1b[35m";
            else:                
                return "\x1b[45m";
        return ''
    
    def cyan(self, fg = True):         
        '''Set the foreground or background color to cyan. 
        The foreground color will be set if "fg" tests True. The background color will be set if "fg" tests False.'''
        if self.enabled:         
            if fg:               
                return "\x1b[36m";
            else:                
                return "\x1b[46m";
        return ''
    
    def white(self, fg = True):        
        '''Set the foreground or background color to white. 
        The foreground color will be set if "fg" tests True. The background color will be set if "fg" tests False.'''
        if self.enabled:         
            if fg:               
                return "\x1b[37m";
            else:                
                return "\x1b[47m";
        return ''
    
    def default(self, fg = True):      
        '''Set the foreground or background color to the default. 
        The foreground color will be set if "fg" tests True. The background color will be set if "fg" tests False.'''
        if self.enabled:         
            if fg:               
                return "\x1b[39m";
            else:                
                return "\x1b[49m";
        return ''


def start_gdb_log(debugger, command, result, dict):
    '''Start logging GDB remote packets by enabling logging with timestamps and 
    thread safe logging. Follow a call to this function with a call to "stop_gdb_log"
    in order to dump out the commands.'''
    global g_log_file
    command_args = shlex.split(command)
    usage = "usage: start_gdb_log [options] [<LOGFILEPATH>]"
    description='''The command enables GDB remote packet logging with timestamps. The packets will be logged to <LOGFILEPATH> if supplied, or a temporary file will be used. Logging stops when stop_gdb_log is called and the packet times will
    be aggregated and displayed.'''
    parser = optparse.OptionParser(description=description, prog='start_gdb_log',usage=usage)
    parser.add_option('-v', '--verbose', action='store_true', dest='verbose', help='display verbose debug info', default=False)
    try:
        (options, args) = parser.parse_args(command_args)
    except:
        return

    if g_log_file:
        result.PutCString ('error: logging is already in progress with file "%s"', g_log_file)
    else:
        args_len = len(args)
        if args_len == 0:
            g_log_file = tempfile.mktemp()
        elif len(args) == 1:
            g_log_file = args[0]

        if g_log_file:
            debugger.HandleCommand('log enable --threadsafe --timestamp --file "%s" gdb-remote packets' % g_log_file);
            result.PutCString ("GDB packet logging enable with log file '%s'\nUse the 'stop_gdb_log' command to stop logging and show packet statistics." % g_log_file)
            return

        result.PutCString ('error: invalid log file path')
    result.PutCString (usage)

def stop_gdb_log(debugger, command, result, dict):
    '''Stop logging GDB remote packets to the file that was specified in a call
    to "start_gdb_log" and normalize the timestamps to be relative to the first
    timestamp in the log file. Also print out statistics for how long each
    command took to allow performance bottlenecks to be determined.'''
    global g_log_file
    # Any commands whose names might be followed by more valid C identifier 
    # characters must be listed here
    command_args = shlex.split(command)
    usage = "usage: stop_gdb_log [options]"
    description='''The command stops a previously enabled GDB remote packet logging command. Packet logging must have been previously enabled with a call to start_gdb_log.'''
    parser = optparse.OptionParser(description=description, prog='stop_gdb_log',usage=usage)
    parser.add_option('-v', '--verbose', action='store_true', dest='verbose', help='display verbose debug info', default=False)
    parser.add_option('-q', '--quiet', action='store_true', dest='quiet', help='display verbose debug info', default=False)
    parser.add_option('-C', '--color', action='store_true', dest='color', help='add terminal colors', default=False)
    parser.add_option('-c', '--sort-by-count', action='store_true', dest='sort_count', help='display verbose debug info', default=False)
    parser.add_option('-s', '--symbolicate', action='store_true', dest='symbolicate', help='symbolicate addresses in log using current "lldb.target"', default=False)
    try:
        (options, args) = parser.parse_args(command_args)
    except:
        return
    options.colors = TerminalColors(options.color)
    options.symbolicator = None
    if options.symbolicate:
        if lldb.target:
            import lldb.utils.symbolication
            options.symbolicator = lldb.utils.symbolication.Symbolicator()
            options.symbolicator.target = lldb.target
        else:
            print "error: can't symbolicate without a target"

    if not g_log_file:
        result.PutCString ('error: logging must have been previously enabled with a call to "stop_gdb_log"')
    elif os.path.exists (g_log_file):
        if len(args) == 0:
            debugger.HandleCommand('log disable gdb-remote packets');
            result.PutCString ("GDB packet logging disabled. Logged packets are in '%s'" % g_log_file)
            parse_gdb_log_file (g_log_file, options)
        else:
            result.PutCString (usage)
    else:
        print 'error: the GDB packet log file "%s" does not exist' % g_log_file

def is_hex_byte(str):
    if len(str) == 2:
        return str[0] in string.hexdigits and str[1] in string.hexdigits;
    return False

# global register info list
g_register_infos = list()
g_max_register_info_name_len = 0

class RegisterInfo:
    """Class that represents register information"""
    def __init__(self, kvp):
        self.info = dict()
        for kv in kvp:
            key = kv[0]
            value = kv[1]
            self.info[key] = value
    def name(self):
        '''Get the name of the register.'''
        if self.info and 'name' in self.info:
            return self.info['name']
        return None

    def bit_size(self):
        '''Get the size in bits of the register.'''
        if self.info and 'bitsize' in self.info:
            return int(self.info['bitsize'])
        return 0

    def byte_size(self):
        '''Get the size in bytes of the register.'''
        return self.bit_size() / 8

    def get_value_from_hex_string(self, hex_str):
        '''Dump the register value given a native byte order encoded hex ASCII byte string.'''
        encoding = self.info['encoding']
        bit_size = self.bit_size()
        packet = Packet(hex_str)
        if encoding == 'uint':
            uval = packet.get_hex_uint(g_byte_order)
            if bit_size == 8:
                return '0x%2.2x' % (uval)
            elif bit_size == 16:
                return '0x%4.4x' % (uval)
            elif bit_size == 32:
                return '0x%8.8x' % (uval)
            elif bit_size == 64:
                return '0x%16.16x' % (uval)
        bytes = list();
        uval = packet.get_hex_uint8()
        while uval != None:
            bytes.append(uval)
            uval = packet.get_hex_uint8()
        value_str = '0x'
        if g_byte_order == 'little':
            bytes.reverse()
        for byte in bytes:
            value_str += '%2.2x' % byte
        return '%s' % (value_str)
    
    def __str__(self):
        '''Dump the register info key/value pairs'''
        s = ''
        for key in self.info.keys():
            if s:
                s += ', '
            s += "%s=%s " % (key, self.info[key])
        return s
    
class Packet:
    """Class that represents a packet that contains string data"""
    def __init__(self, packet_str):
        self.str = packet_str
        
    def peek_char(self):
        ch = 0
        if self.str:
            ch = self.str[0]
        return ch
        
    def get_char(self):
        ch = 0
        if self.str:
            ch = self.str[0]
            self.str = self.str[1:]
        return ch
        
    def get_hex_uint8(self):
        if self.str and len(self.str) >= 2 and self.str[0] in string.hexdigits and self.str[1] in string.hexdigits:
            uval = int(self.str[0:2], 16)
            self.str = self.str[2:]
            return uval
        return None
        
    def get_hex_uint16(self, byte_order):
        uval = 0
        if byte_order == 'big':
            uval |= self.get_hex_uint8() << 8
            uval |= self.get_hex_uint8()
        else:
            uval |= self.get_hex_uint8()
            uval |= self.get_hex_uint8() << 8
        return uval
        
    def get_hex_uint32(self, byte_order):
        uval = 0
        if byte_order == 'big':
            uval |= self.get_hex_uint8() << 24
            uval |= self.get_hex_uint8() << 16
            uval |= self.get_hex_uint8() << 8
            uval |= self.get_hex_uint8()
        else:
            uval |= self.get_hex_uint8()
            uval |= self.get_hex_uint8() << 8
            uval |= self.get_hex_uint8() << 16
            uval |= self.get_hex_uint8() << 24
        return uval
        
    def get_hex_uint64(self, byte_order):
        uval = 0
        if byte_order == 'big':
            uval |= self.get_hex_uint8() << 56
            uval |= self.get_hex_uint8() << 48
            uval |= self.get_hex_uint8() << 40
            uval |= self.get_hex_uint8() << 32
            uval |= self.get_hex_uint8() << 24
            uval |= self.get_hex_uint8() << 16
            uval |= self.get_hex_uint8() << 8
            uval |= self.get_hex_uint8()
        else:
            uval |= self.get_hex_uint8()
            uval |= self.get_hex_uint8() << 8
            uval |= self.get_hex_uint8() << 16
            uval |= self.get_hex_uint8() << 24
            uval |= self.get_hex_uint8() << 32
            uval |= self.get_hex_uint8() << 40
            uval |= self.get_hex_uint8() << 48
            uval |= self.get_hex_uint8() << 56
        return uval
    
    def get_hex_chars(self, n = 0):
        str_len = len(self.str)
        if n == 0:
            # n was zero, so we need to determine all hex chars and 
            # stop when we hit the end of the string of a non-hex character
            while n < str_len and self.str[n] in string.hexdigits:
                n = n + 1
        else:
            if n > str_len:
                return None # Not enough chars
            # Verify all chars are hex if a length was specified
            for i in range(n):
                if self.str[i] not in string.hexdigits:
                    return None # Not all hex digits
        if n == 0:
            return None
        hex_str = self.str[0:n]
        self.str = self.str[n:]
        return hex_str
        
    def get_hex_uint(self, byte_order, n = 0):
        if byte_order == 'big':
            hex_str = self.get_hex_chars(n)
            if hex_str == None:
                return None
            return int(hex_str, 16)
        else:
            uval = self.get_hex_uint8()
            if uval == None:
                return None
            uval_result = 0
            shift = 0
            while uval != None:
                uval_result |= (uval << shift)
                shift += 8
                uval = self.get_hex_uint8()
            return uval_result
        
    def get_key_value_pairs(self):
        kvp = list()
        key_value_pairs = string.split(self.str, ';')
        for key_value_pair in key_value_pairs:
            if len(key_value_pair):
                kvp.append(string.split(key_value_pair, ':'))
        return kvp

    def split(self, ch):
        return string.split(self.str, ch)

    def split_hex(self, ch, byte_order):
        hex_values = list()
        strings = string.split(self.str, ch)
        for str in strings:
            hex_values.append(Packet(str).get_hex_uint(byte_order))
        return hex_values
    
    def __str__(self):
        return self.str
    
    def __len__(self):
        return len(self.str)

g_thread_suffix_regex = re.compile(';thread:([0-9a-fA-F]+);')
def get_thread_from_thread_suffix(str):
    if str:
        match = g_thread_suffix_regex.match (str)
        if match:
            return int(match.group(1), 16)
    return None

def cmd_stop_reply(options, cmd, args):
    print "get_last_stop_info()"

def rsp_stop_reply(options, cmd, cmd_args, rsp):
    global g_byte_order
    packet = Packet(rsp)
    stop_type = packet.get_char()
    if stop_type == 'T' or stop_type == 'S':
        signo  = packet.get_hex_uint8()
        print '    signal = %i' % signo
        key_value_pairs = packet.get_key_value_pairs()
        for key_value_pair in key_value_pairs:
            key = key_value_pair[0]
            value = key_value_pair[1]
            if is_hex_byte(key):
                reg_num = Packet(key).get_hex_uint8()
                print '    ' + get_register_name_equal_value (options, reg_num, value)
            else:
                print '    %s = %s' % (key, value)
    elif stop_type == 'W':
        exit_status = packet.get_hex_uint8()
        print 'exit (status=%i)' % exit_status
    elif stop_type == 'O':
        print 'stdout = %s' % packet.str
        

def cmd_unknown_packet(options, cmd, args):
    if args:
        print "cmd: %s, args: %s", cmd, args
    else:
        print "cmd: %s", cmd

def cmd_query_packet(options, cmd, args):
    if args:
        print "query: %s, args: %s" % (cmd, args)
    else:
        print "query: %s" % (cmd)

def rsp_ok_error(rsp):
    print "rsp: ", rsp

def rsp_ok_means_supported(options, cmd, cmd_args, rsp):
    if rsp == 'OK':
        print "%s%s is supported" % (cmd, cmd_args)
    elif rsp == '':
        print "%s%s is not supported" % (cmd, cmd_args)
    else:
        print "%s%s -> %s" % (cmd, cmd_args, rsp)

def rsp_ok_means_success(options, cmd, cmd_args, rsp):
    if rsp == 'OK':
        print "success"
    elif rsp == '':
        print "%s%s is not supported" % (cmd, cmd_args)
    else:
        print "%s%s -> %s" % (cmd, cmd_args, rsp)

def rsp_dump_key_value_pairs(options, cmd, cmd_args, rsp):
    if rsp:
        packet = Packet(rsp)
        key_value_pairs = packet.get_key_value_pairs()
        for key_value_pair in key_value_pairs:
            key = key_value_pair[0]
            value = key_value_pair[1]
            print "    %s = %s" % (key, value)
    else:
        print "not supported"

def cmd_vCont(options, cmd, args):
    if args == '?':
        print "%s: get supported extended continue modes" % (cmd)
    else:
        got_other_threads = 0
        s = ''
        for thread_action in string.split(args[1:], ';'):
            (short_action, thread) = string.split(thread_action, ':')
            tid = int(thread, 16)
            if short_action == 'c':
                action = 'continue'
            elif short_action == 's':
                action = 'step'
            elif short_action[0] == 'C':
                action = 'continue with signal 0x%s' % (short_action[1:])
            elif short_action == 'S':
                action = 'step with signal 0x%s' % (short_action[1:])
            else:
                action = short_action
            if s:
                s += ', '
            if tid == -1:
                got_other_threads = 1
                s += 'other-threads:'
            else:
                s += 'thread 0x%4.4x: %s' % (tid, action)
        if got_other_threads:
            print "extended_continue (%s)" % (s)
        else:
            print "extended_continue (%s, other-threads: suspend)" % (s)

def rsp_vCont(options, cmd, cmd_args, rsp):
    if cmd_args == '?':
        # Skip the leading 'vCont;'
        rsp = rsp[6:]
        modes = string.split(rsp, ';')
        s = "%s: supported extended continue modes include: " % (cmd)
        
        for i, mode in enumerate(modes):
            if i: 
                s += ', '
            if mode == 'c':
                s += 'continue'
            elif mode == 'C':
                s += 'continue with signal'
            elif mode == 's':
                s += 'step'
            elif mode == 'S':
                s += 'step with signal'
            else:
                s += 'unrecognized vCont mode: ', mode
        print s
    elif rsp:
        if rsp[0] == 'T' or rsp[0] == 'S' or rsp[0] == 'W' or rsp[0] == 'X':
            rsp_stop_reply (options, cmd, cmd_args, rsp)
            return
        if rsp[0] == 'O':
            print "stdout: %s" % (rsp)
            return
    else:
        print "not supported (cmd = '%s', args = '%s', rsp = '%s')" % (cmd, cmd_args, rsp)

def cmd_vAttach(options, cmd, args):
    (extra_command, args) = string.split(args, ';')
    if extra_command:
        print "%s%s(%s)" % (cmd, extra_command, args)
    else:
        print "attach_pid(%s)" % args

def cmd_qRegisterInfo(options, cmd, args):
    print 'query_register_info(reg_num=%i)' % (int(args, 16))

def rsp_qRegisterInfo(options, cmd, cmd_args, rsp):
    global g_max_register_info_name_len
    print 'query_register_info(reg_num=%i):' % (int(cmd_args, 16)),
    if len(rsp) == 3 and rsp[0] == 'E':
        g_max_register_info_name_len = 0
        for reg_info in g_register_infos:
            name_len = len(reg_info.name())
            if g_max_register_info_name_len < name_len:
                g_max_register_info_name_len = name_len
        print' DONE'
    else:
        packet = Packet(rsp)
        reg_info = RegisterInfo(packet.get_key_value_pairs())
        g_register_infos.append(reg_info)
        print reg_info
        

def cmd_qThreadInfo(options, cmd, args):
    if cmd == 'qfThreadInfo':
        query_type = 'first'
    else: 
        query_type = 'subsequent'
    print 'get_current_thread_list(type=%s)' % (query_type)

def rsp_qThreadInfo(options, cmd, cmd_args, rsp):
    packet = Packet(rsp)
    response_type = packet.get_char()
    if response_type == 'm':
        tids = packet.split_hex(';', 'big')
        for i, tid in enumerate(tids):
            if i:
                print ',',
            print '0x%x' % (tid),
        print
    elif response_type == 'l':
        print 'END'

def rsp_hex_big_endian(options, cmd, cmd_args, rsp):
    packet = Packet(rsp)
    uval = packet.get_hex_uint('big')
    print '%s: 0x%x' % (cmd, uval)

def cmd_read_memory(options, cmd, args):
    packet = Packet(args)
    addr = packet.get_hex_uint('big')
    comma = packet.get_char()
    size = packet.get_hex_uint('big')
    print 'read_memory (addr = 0x%x, size = %u)' % (addr, size)

def dump_hex_memory_buffer(addr, hex_byte_str):
    packet = Packet(hex_byte_str)
    idx = 0
    ascii = ''
    uval = packet.get_hex_uint8()
    while uval != None:
        if ((idx % 16) == 0):
            if ascii:
                print '  ', ascii
                ascii = ''
            print '0x%x:' % (addr + idx),
        print '%2.2x' % (uval),
        if 0x20 <= uval and uval < 0x7f:
            ascii += '%c' % uval
        else:
            ascii += '.'
        uval = packet.get_hex_uint8()
        idx = idx + 1
    if ascii:
        print '  ', ascii
        ascii = ''        
    
def cmd_write_memory(options, cmd, args):
    packet = Packet(args)
    addr = packet.get_hex_uint('big')
    if packet.get_char() != ',':
        print 'error: invalid write memory command (missing comma after address)'
        return
    size = packet.get_hex_uint('big')
    if packet.get_char() != ':':
        print 'error: invalid write memory command (missing colon after size)'
        return
    print 'write_memory (addr = 0x%x, size = %u, data:' % (addr, size)
    dump_hex_memory_buffer (addr, packet.str) 

def cmd_alloc_memory(options, cmd, args):
    packet = Packet(args)
    byte_size = packet.get_hex_uint('big')
    if packet.get_char() != ',':
        print 'error: invalid allocate memory command (missing comma after address)'
        return
    print 'allocate_memory (byte-size = %u (0x%x), permissions = %s)' % (byte_size, byte_size, packet.str)

def rsp_alloc_memory(options, cmd, cmd_args, rsp):
    packet = Packet(rsp)
    addr = packet.get_hex_uint('big')
    print 'addr = 0x%x' % addr

def cmd_dealloc_memory(options, cmd, args):
    packet = Packet(args)
    addr = packet.get_hex_uint('big')
    if packet.get_char() != ',':
        print 'error: invalid allocate memory command (missing comma after address)'
        return
    print 'deallocate_memory (addr = 0x%x, permissions = %s)' % (addr, packet.str)

def rsp_memory_bytes(options, cmd, cmd_args, rsp):
    addr = Packet(cmd_args).get_hex_uint('big')
    dump_hex_memory_buffer (addr, rsp) 

def get_register_name_equal_value(options, reg_num, hex_value_str):
    if reg_num < len(g_register_infos):
        reg_info = g_register_infos[reg_num]
        value_str = reg_info.get_value_from_hex_string (hex_value_str)
        s = reg_info.name() + ' = '
        if options.symbolicator:
            symbolicated_addresses = options.symbolicator.symbolicate (int(value_str, 0))
            if symbolicated_addresses:
                s += options.colors.magenta()
                s += '%s' % symbolicated_addresses[0]
                s += options.colors.reset()
                return s
        s += value_str
        return s
    else:
        reg_value = Packet(hex_value_str).get_hex_uint(g_byte_order)
        return 'reg(%u) = 0x%x' % (reg_num, reg_value)

def cmd_read_one_reg(options, cmd, args):
    packet = Packet(args)
    reg_num = packet.get_hex_uint('big')
    tid = get_thread_from_thread_suffix (packet.str)
    name = None
    if reg_num < len(g_register_infos):
        name = g_register_infos[reg_num].name ()
    if packet.str:
        packet.get_char() # skip ;
        thread_info = packet.get_key_value_pairs()
        tid = int(thread_info[0][1], 16)
    s = 'read_register (reg_num=%u' % reg_num
    if name:
        s += ' (%s)' % (name)
    if tid != None:
        s += ', tid = 0x%4.4x' % (tid)
    s += ')'
    print s

def rsp_read_one_reg(options, cmd, cmd_args, rsp):
    packet = Packet(cmd_args)
    reg_num = packet.get_hex_uint('big')
    print get_register_name_equal_value (options, reg_num, rsp)

def cmd_write_one_reg(options, cmd, args):
    packet = Packet(args)
    reg_num = packet.get_hex_uint('big')
    if packet.get_char() != '=':
        print 'error: invalid register write packet'
    else:
        name = None
        hex_value_str = packet.get_hex_chars()
        tid = get_thread_from_thread_suffix (packet.str)
        s = 'write_register (reg_num=%u' % reg_num
        if name:
            s += ' (%s)' % (name)
        s += ', value = '
        s += get_register_name_equal_value(options, reg_num, hex_value_str)
        if tid != None:
            s += ', tid = 0x%4.4x' % (tid)
        s += ')'
        print s

def dump_all_regs(packet):
    for reg_info in g_register_infos:
        nibble_size = reg_info.bit_size() / 4
        hex_value_str = packet.get_hex_chars(nibble_size)
        if hex_value_str != None:
            value = reg_info.get_value_from_hex_string (hex_value_str)
            print '%*s = %s' % (g_max_register_info_name_len, reg_info.name(), value)
        else:
            return
    
def cmd_read_all_regs(cmd, cmd_args):
    packet = Packet(cmd_args)
    packet.get_char() # toss the 'g' command character
    tid = get_thread_from_thread_suffix (packet.str)
    if tid != None:
        print 'read_all_register(thread = 0x%4.4x)' % tid
    else:
        print 'read_all_register()'

def rsp_read_all_regs(options, cmd, cmd_args, rsp):
    packet = Packet(rsp)
    dump_all_regs (packet)

def cmd_write_all_regs(options, cmd, args):
    packet = Packet(args)
    print 'write_all_registers()'
    dump_all_regs (packet)
    
g_bp_types = [ "software_bp", "hardware_bp", "write_wp", "read_wp", "access_wp" ]

def cmd_bp(options, cmd, args):
    if cmd == 'Z':
        s = 'set_'
    else:
        s = 'clear_'
    packet = Packet (args)
    bp_type = packet.get_hex_uint('big')
    packet.get_char() # Skip ,
    bp_addr = packet.get_hex_uint('big')
    packet.get_char() # Skip ,
    bp_size = packet.get_hex_uint('big')
    s += g_bp_types[bp_type]
    s += " (addr = 0x%x, size = %u)" % (bp_addr, bp_size)
    print s

def cmd_mem_rgn_info(options, cmd, args):
    packet = Packet(args)
    packet.get_char() # skip ':' character
    addr = packet.get_hex_uint('big')
    print 'get_memory_region_info (addr=0x%x)' % (addr)

def cmd_kill(options, cmd, args):
    print 'kill_process()'

gdb_remote_commands = {
    '\\?'                     : { 'cmd' : cmd_stop_reply    , 'rsp' : rsp_stop_reply          , 'name' : "stop reply pacpket"},
    'QStartNoAckMode'         : { 'cmd' : cmd_query_packet  , 'rsp' : rsp_ok_means_supported  , 'name' : "query if no ack mode is supported"},
    'QThreadSuffixSupported'  : { 'cmd' : cmd_query_packet  , 'rsp' : rsp_ok_means_supported  , 'name' : "query if thread suffix is supported" },
    'QListThreadsInStopReply' : { 'cmd' : cmd_query_packet  , 'rsp' : rsp_ok_means_supported  , 'name' : "query if threads in stop reply packets are supported" },
    'qVAttachOrWaitSupported' : { 'cmd' : cmd_query_packet  , 'rsp' : rsp_ok_means_supported  , 'name' : "query if threads attach with optional wait is supported" },
    'qHostInfo'               : { 'cmd' : cmd_query_packet  , 'rsp' : rsp_dump_key_value_pairs, 'name' : "get host information" },
    'vCont'                   : { 'cmd' : cmd_vCont         , 'rsp' : rsp_vCont               , 'name' : "extended continue command" },
    'vAttach'                 : { 'cmd' : cmd_vAttach       , 'rsp' : rsp_stop_reply          , 'name' : "attach to process" },
    'qRegisterInfo'           : { 'cmd' : cmd_qRegisterInfo , 'rsp' : rsp_qRegisterInfo       , 'name' : "query register info" },
    'qfThreadInfo'            : { 'cmd' : cmd_qThreadInfo   , 'rsp' : rsp_qThreadInfo         , 'name' : "get current thread list" },
    'qsThreadInfo'            : { 'cmd' : cmd_qThreadInfo   , 'rsp' : rsp_qThreadInfo         , 'name' : "get current thread list" },
    'qShlibInfoAddr'          : { 'cmd' : cmd_query_packet  , 'rsp' : rsp_hex_big_endian      , 'name' : "get shared library info address" },
    'qMemoryRegionInfo'       : { 'cmd' : cmd_mem_rgn_info  , 'rsp' : rsp_dump_key_value_pairs, 'name' : "get memory region information" },
    'm'                       : { 'cmd' : cmd_read_memory   , 'rsp' : rsp_memory_bytes        , 'name' : "read memory" },
    'M'                       : { 'cmd' : cmd_write_memory  , 'rsp' : rsp_ok_means_success    , 'name' : "write memory" },
    '_M'                      : { 'cmd' : cmd_alloc_memory  , 'rsp' : rsp_alloc_memory        , 'name' : "allocate memory" },
    '_m'                      : { 'cmd' : cmd_dealloc_memory, 'rsp' : rsp_ok_means_success    , 'name' : "deallocate memory" },
    'p'                       : { 'cmd' : cmd_read_one_reg  , 'rsp' : rsp_read_one_reg        , 'name' : "read single register" },
    'P'                       : { 'cmd' : cmd_write_one_reg , 'rsp' : rsp_ok_means_success    , 'name' : "write single register" },
    'g'                       : { 'cmd' : cmd_read_all_regs , 'rsp' : rsp_read_all_regs       , 'name' : "read all registers" },
    'G'                       : { 'cmd' : cmd_write_all_regs, 'rsp' : rsp_ok_means_success    , 'name' : "write all registers" },
    'z'                       : { 'cmd' : cmd_bp            , 'rsp' : rsp_ok_means_success    , 'name' : "clear breakpoint or watchpoint" },
    'Z'                       : { 'cmd' : cmd_bp            , 'rsp' : rsp_ok_means_success    , 'name' : "set breakpoint or watchpoint" },
    'k'                       : { 'cmd' : cmd_kill          , 'rsp' : rsp_stop_reply          , 'name' : "kill process" },
}
def parse_gdb_log_file(file, options):
    '''Parse a GDB log file that was generated by enabling logging with:
    (lldb) log enable --threadsafe --timestamp --file <FILE> gdb-remote packets
    This log file will contain timestamps and this fucntion will then normalize
    those packets to be relative to the first value timestamp that is found and
    show delta times between log lines and also keep track of how long it takes
    for GDB remote commands to make a send/receive round trip. This can be
    handy when trying to figure out why some operation in the debugger is taking
    a long time during a preset set of debugger commands.'''

    tricky_commands = [ 'qRegisterInfo' ]
    timestamp_regex = re.compile('(\s*)([1-9][0-9]+\.[0-9]+)([^0-9].*)$')
    packet_name_regex = re.compile('([A-Za-z_]+)[^a-z]')
    packet_transmit_name_regex = re.compile('(?P<direction>send|read) packet: (?P<packet>.*)')
    packet_contents_name_regex = re.compile('\$([^#]+)#[0-9a-fA-F]{2}')
    packet_names_regex_str = '(' + '|'.join(gdb_remote_commands.keys()) + ')(.*)';
    packet_names_regex = re.compile(packet_names_regex_str);
    
    base_time = 0.0
    last_time = 0.0
    packet_send_time = 0.0
    packet_total_times = {}
    packet_count = {}
    file = open(file)
    lines = file.read().splitlines()
    last_command = None
    last_command_args = None
    last_command_packet = None
    for line in lines:
        packet_name = None
        m = packet_transmit_name_regex.search(line)
        is_command = False
        if m:
            direction = m.group('direction')
            is_command = direction == 'send'
            packet = m.group('packet')
            sys.stdout.write(options.colors.green())
            if options.quiet:
                if is_command:
                    print '-->', packet
                else:
                    print '<--', packet
            else:
                print '#  ', line
            sys.stdout.write(options.colors.reset())
                
            #print 'direction = "%s", packet = "%s"' % (direction, packet)
            
            if packet[0] == '+':
                print 'ACK'
            elif packet[0] == '-':
                print 'NACK'
            elif packet[0] == '$':
                m = packet_contents_name_regex.match(packet)
                if m:
                    contents = m.group(1)
                    if is_command:
                        m = packet_names_regex.match (contents)
                        if m:
                            last_command = m.group(1)
                            packet_name = last_command
                            last_command_args = m.group(2)
                            last_command_packet = contents
                            gdb_remote_commands[last_command]['cmd'](options, last_command, last_command_args)
                        else:
                            packet_match = packet_name_regex.match (line[idx+14:])
                            if packet_match:
                                packet_name = packet_match.group(1)
                                for tricky_cmd in tricky_commands:
                                    if packet_name.find (tricky_cmd) == 0:
                                        packet_name = tricky_cmd
                            else:
                                packet_name = contents
                            last_command = None
                            last_command_args = None
                            last_command_packet = None
                    elif last_command:
                        gdb_remote_commands[last_command]['rsp'](options, last_command, last_command_args, contents)
                else:
                    print 'error: invalid packet: "', packet, '"'
            else:
                print '???'
        else:
            print '## ', line
        match = timestamp_regex.match (line)
        if match:
            curr_time = float (match.group(2))
            delta = 0.0
            if base_time:
                delta = curr_time - last_time
            else:
                base_time = curr_time
            
            if is_command:
                packet_send_time = curr_time
            elif line.find('read packet: $') >= 0 and packet_name:
                if packet_name in packet_total_times:
                    packet_total_times[packet_name] += delta
                    packet_count[packet_name] += 1
                else:
                    packet_total_times[packet_name] = delta
                    packet_count[packet_name] = 1
                packet_name = None

            if not options or not options.quiet:
                print '%s%.6f %+.6f%s' % (match.group(1), curr_time - base_time, delta, match.group(3))
            last_time = curr_time
        # else:
        #     print line
    if packet_total_times:
        total_packet_time = 0.0
        total_packet_count = 0
        for key, vvv in packet_total_times.items():
            # print '  key = (%s) "%s"' % (type(key), key)
            # print 'value = (%s) %s' % (type(vvv), vvv)
            # if type(vvv) == 'float':
            total_packet_time += vvv
        for key, vvv in packet_count.items():
            total_packet_count += vvv

        print '#---------------------------------------------------'
        print '# Packet timing summary:'
        print '# Totals: time - %6f count %6d' % (total_packet_time, total_packet_count)
        print '#---------------------------------------------------'
        print '# Packet                   Time (sec) Percent Count '
        print '#------------------------- ---------- ------- ------'
        if options and options.sort_count:
            res = sorted(packet_count, key=packet_count.__getitem__, reverse=True)
        else:
            res = sorted(packet_total_times, key=packet_total_times.__getitem__, reverse=True)

        if last_time > 0.0:
            for item in res:
                packet_total_time = packet_total_times[item]
                packet_percent = (packet_total_time / total_packet_time)*100.0
                if packet_percent >= 10.0:
                    print "  %24s %.6f   %.2f%% %6d" % (item, packet_total_time, packet_percent, packet_count[item])
                else:
                    print "  %24s %.6f   %.2f%%  %6d" % (item, packet_total_time, packet_percent, packet_count[item])
                    
    
    
if __name__ == '__main__':
    usage = "usage: gdbremote [options]"
    description='''The command disassembles a GDB remote packet log.'''
    parser = optparse.OptionParser(description=description, prog='gdbremote',usage=usage)
    parser.add_option('-v', '--verbose', action='store_true', dest='verbose', help='display verbose debug info', default=False)
    parser.add_option('-q', '--quiet', action='store_true', dest='quiet', help='display verbose debug info', default=False)
    parser.add_option('-C', '--color', action='store_true', dest='color', help='add terminal colors', default=False)
    parser.add_option('-c', '--sort-by-count', action='store_true', dest='sort_count', help='display verbose debug info', default=False)
    parser.add_option('--crashlog', type='string', dest='crashlog', help='symbolicate using a darwin crash log file', default=False)
    try:
        (options, args) = parser.parse_args(sys.argv[1:])
    except:
        print 'error: argument error'
        sys.exit(1)

    options.colors = TerminalColors(options.color)
    options.symbolicator = None
    if options.crashlog:
        import lldb
        lldb.debugger = lldb.SBDebugger.Create()
        import lldb.macosx.crashlog
        options.symbolicator = lldb.macosx.crashlog.CrashLog(options.crashlog)
        print '%s' % (options.symbolicator)

    # This script is being run from the command line, create a debugger in case we are
    # going to use any debugger functions in our function.
    for file in args:
        print '#----------------------------------------------------------------------'
        print "# GDB remote log file: '%s'" % file
        print '#----------------------------------------------------------------------'
        parse_gdb_log_file (file, options)
    if options.symbolicator:
        print '%s' % (options.symbolicator)
        
else:
    import lldb
    if lldb.debugger:    
        # This initializer is being run from LLDB in the embedded command interpreter
        # Add any commands contained in this module to LLDB
        lldb.debugger.HandleCommand('command script add -f gdbremote.start_gdb_log start_gdb_log')
        lldb.debugger.HandleCommand('command script add -f gdbremote.stop_gdb_log stop_gdb_log')
        print 'The "start_gdb_log" and "stop_gdb_log" commands are now installed and ready for use, type "start_gdb_log --help" or "stop_gdb_log --help" for more information'
