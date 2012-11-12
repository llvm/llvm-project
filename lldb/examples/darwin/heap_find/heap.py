#!/usr/bin/python

#----------------------------------------------------------------------
# This module is designed to live inside the "lldb" python package
# in the "lldb.macosx" package. To use this in the embedded python
# interpreter using "lldb" just import it:
#
#   (lldb) script import lldb.macosx.heap
#----------------------------------------------------------------------

import lldb
import commands
import optparse
import os
import os.path
import re
import shlex
import string
import tempfile
import lldb.utils.symbolication

g_libheap_dylib_dir = None
g_libheap_dylib_dict = dict()
g_verbose = False

def load_dylib():
    if lldb.target:
        global g_libheap_dylib_dir
        global g_libheap_dylib_dict
        triple = lldb.target.triple
        if triple in g_libheap_dylib_dict:
            libheap_dylib_path = g_libheap_dylib_dict[triple]
        else:
            if not g_libheap_dylib_dir:
                g_libheap_dylib_dir = tempfile.gettempdir() + '/lldb-dylibs'
            triple_dir = g_libheap_dylib_dir + '/' + triple + '/' + __name__
            if not os.path.exists(triple_dir):
                os.makedirs(triple_dir)
            libheap_dylib_path = triple_dir + '/libheap.dylib'
            g_libheap_dylib_dict[triple] = libheap_dylib_path
        heap_code_directory = os.path.dirname(__file__) + '/heap'
        heap_source_file = heap_code_directory + '/heap_find.cpp'
        # Check if the dylib doesn't exist, or if "heap_find.cpp" is newer than the dylib
        if not os.path.exists(libheap_dylib_path) or os.stat(heap_source_file).st_mtime > os.stat(libheap_dylib_path).st_mtime:
            # Remake the dylib
            make_command = '(cd "%s" ; make EXE="%s" ARCH=%s)' % (heap_code_directory, libheap_dylib_path, string.split(triple, '-')[0])
            (make_exit_status, make_output) = commands.getstatusoutput(make_command)
            if make_exit_status != 0:
                return 'error: make failed: %s' % (make_output)
        if os.path.exists(libheap_dylib_path):
            libheap_dylib_spec = lldb.SBFileSpec(libheap_dylib_path)
            if lldb.target.FindModule(libheap_dylib_spec):
                return None # success, 'libheap.dylib' already loaded
            if lldb.process:
                state = lldb.process.state
                if state == lldb.eStateStopped:
                    (libheap_dylib_path)
                    error = lldb.SBError()
                    image_idx = lldb.process.LoadImage(libheap_dylib_spec, error)
                    if error.Success():
                        return None
                    else:
                        if error:
                            return 'error: %s' % error
                        else:
                            return 'error: "process load \'%s\'" failed' % libheap_dylib_spec
                else:
                    return 'error: process is not stopped'
            else:
                return 'error: invalid process'
        else:
            return 'error: file does not exist "%s"' % libheap_dylib_path
    else:
        return 'error: invalid target'
        
    debugger.HandleCommand('process load "%s"' % libheap_dylib_path)
    if lldb.target.FindModule(libheap_dylib_spec):
        return None # success, 'libheap.dylib' already loaded
    return 'error: failed to load "%s"' % libheap_dylib_path

def get_member_types_for_offset(value_type, offset, member_list):
    member = value_type.GetFieldAtIndex(0)
    search_bases = False
    if member:
        if member.GetOffsetInBytes() <= offset:
            for field_idx in range (value_type.GetNumberOfFields()):
                member = value_type.GetFieldAtIndex(field_idx)
                member_byte_offset = member.GetOffsetInBytes()
                member_end_byte_offset = member_byte_offset + member.type.size
                if member_byte_offset <= offset and offset < member_end_byte_offset:
                    member_list.append(member)
                    get_member_types_for_offset (member.type, offset - member_byte_offset, member_list)
                    return
        else:
            search_bases = True
    else:
        search_bases = True
    if search_bases:
        for field_idx in range (value_type.GetNumberOfDirectBaseClasses()):
            member = value_type.GetDirectBaseClassAtIndex(field_idx)
            member_byte_offset = member.GetOffsetInBytes()
            member_end_byte_offset = member_byte_offset + member.type.size
            if member_byte_offset <= offset and offset < member_end_byte_offset:
                member_list.append(member)
                get_member_types_for_offset (member.type, offset - member_byte_offset, member_list)
                return
        for field_idx in range (value_type.GetNumberOfVirtualBaseClasses()):
            member = value_type.GetVirtualBaseClassAtIndex(field_idx)
            member_byte_offset = member.GetOffsetInBytes()
            member_end_byte_offset = member_byte_offset + member.type.size
            if member_byte_offset <= offset and offset < member_end_byte_offset:
                member_list.append(member)
                get_member_types_for_offset (member.type, offset - member_byte_offset, member_list)
                return

def append_regex_callback(option, opt, value, parser):
    try:
        ivar_regex = re.compile(value)
        parser.values.ivar_regex_blacklist.append(ivar_regex)
    except:
        print 'error: an exception was thrown when compiling the ivar regular expression for "%s"' % value
    
def add_common_options(parser):
    parser.add_option('-v', '--verbose', action='store_true', dest='verbose', help='display verbose debug info', default=False)
    parser.add_option('-t', '--type', action='store_true', dest='print_type', help='print the full value of the type for each matching malloc block', default=False)
    parser.add_option('-o', '--po', action='store_true', dest='print_object_description', help='print the object descriptions for any matches', default=False)
    parser.add_option('-z', '--size', action='store_true', dest='show_size', help='print the allocation size in bytes', default=False)
    parser.add_option('-r', '--range', action='store_true', dest='show_range', help='print the allocation address range instead of just the allocation base address', default=False)
    parser.add_option('-m', '--memory', action='store_true', dest='memory', help='dump the memory for each matching block', default=False)
    parser.add_option('-f', '--format', type='string', dest='format', help='the format to use when dumping memory if --memory is specified', default=None)
    parser.add_option('-I', '--omit-ivar-regex', type='string', action='callback', callback=append_regex_callback, dest='ivar_regex_blacklist', default=[], help='specify one or more regular expressions used to backlist any matches that are in ivars')
    parser.add_option('-s', '--stack', action='store_true', dest='stack', help='gets the stack that allocated each malloc block if MallocStackLogging is enabled', default=False)
    parser.add_option('-S', '--stack-history', action='store_true', dest='stack_history', help='gets the stack history for all allocations whose start address matches each malloc block if MallocStackLogging is enabled', default=False)
    parser.add_option('-M', '--max-matches', type='int', dest='max_matches', help='the maximum number of matches to print', default=256)
    parser.add_option('-O', '--offset', type='int', dest='offset', help='the matching data must be at this offset', default=-1)
    parser.add_option('-V', '--vm-regions', action='store_true', dest='check_vm_regions', help='Also check the VM regions', default=False)

def dump_stack_history_entry(result, stack_history_entry, idx):
    address = int(stack_history_entry.address)
    if address:
        type_flags = int(stack_history_entry.type_flags)
        symbolicator = lldb.utils.symbolication.Symbolicator()
        symbolicator.target = lldb.target
        type_str = ''
        if type_flags == 0:
            type_str = 'free'
        else:
            if type_flags & 2:
                type_str = 'alloc'
            elif type_flags & 4:
                type_str = 'free'
            elif type_flags & 1:
                type_str = 'generic'
            else:
                type_str = hex(type_flags)
        result.AppendMessage('stack[%u]: addr = 0x%x, type=%s, frames:' % (idx, address, type_str))
        frame_idx = 0
        idx = 0
        pc = int(stack_history_entry.frames[idx])
        while pc != 0:
            if pc >= 0x1000:
                frames = symbolicator.symbolicate(pc)
                if frames:
                    for frame in frames:
                        result.AppendMessage('     [%u] %s' % (frame_idx, frame))
                        frame_idx += 1
                else:
                    result.AppendMessage('     [%u] 0x%x' % (frame_idx, pc))
                    frame_idx += 1
                idx = idx + 1
                pc = int(stack_history_entry.frames[idx])
            else:
                pc = 0
        result.AppendMessage('')
            
def dump_stack_history_entries(result, addr, history):
    # malloc_stack_entry *get_stack_history_for_address (const void * addr)
    expr = 'get_stack_history_for_address((void *)0x%x, %u)' % (addr, history)
    expr_sbvalue = lldb.frame.EvaluateExpression (expr)
    if expr_sbvalue.error.Success():
        if expr_sbvalue.unsigned:
            expr_value = lldb.value(expr_sbvalue)  
            idx = 0;
            stack_history_entry = expr_value[idx]
            while int(stack_history_entry.address) != 0:
                dump_stack_history_entry(result, stack_history_entry, idx)
                idx = idx + 1
                stack_history_entry = expr_value[idx]
        else:
            result.AppendMessage('"%s" returned zero' % (expr))
    else:
        result.AppendMessage('error: expression failed "%s" => %s' % (expr, expr_sbvalue.error))
    

def display_match_results (result, options, arg_str_description, expr_sbvalue, print_no_matches = True):
    if expr_sbvalue.error.Success():
        if expr_sbvalue.unsigned:
            match_value = lldb.value(expr_sbvalue)  
            i = 0
            match_idx = 0
            while 1:
                print_entry = True
                match_entry = match_value[i]; i += 1
                if i >= options.max_matches:
                    result.AppendMessage('error: the max number of matches (%u) was reached, use the --max-matches option to get more results' % (options.max_matches))
                    break
                malloc_addr = match_entry.addr.sbvalue.unsigned
                if malloc_addr == 0:
                    break
                malloc_size = int(match_entry.size)
                offset = int(match_entry.offset)
                
                if options.offset >= 0 and options.offset != offset:
                    print_entry = False
                else:                    
                    match_addr = malloc_addr + offset
                    dynamic_value = match_entry.addr.sbvalue.GetDynamicValue(lldb.eDynamicCanRunTarget)
                    description = '%#x: ' % (match_addr)
                    if options.show_size:
                        description += '<%5u> ' % (malloc_size)
                    if options.show_range:
                        if offset > 0:
                            description += '[%#x - %#x) + %-6u ' % (malloc_addr, malloc_addr + malloc_size, offset)
                        else:
                            description += '[%#x - %#x)' % (malloc_addr, malloc_addr + malloc_size)
                    else:
                        if options.type != 'isa':
                            description += '%#x + %-6u ' % (malloc_addr, offset)
                    derefed_dynamic_value = None
                    if dynamic_value.type.name == 'void *':
                        if options.type == 'pointer' and malloc_size == 4096:
                            error = lldb.SBError()
                            data = bytearray(lldb.process.ReadMemory(malloc_addr, 16, error))
                            if data == '\xa1\xa1\xa1\xa1AUTORELEASE!':
                                ptr_size = lldb.target.addr_size
                                thread = lldb.process.ReadUnsignedFromMemory (malloc_addr + 16 + ptr_size, ptr_size, error)
                                #   4 bytes  0xa1a1a1a1
                                #  12 bytes  'AUTORELEASE!'
                                # ptr bytes  autorelease insertion point
                                # ptr bytes  pthread_t
                                # ptr bytes  next colder page
                                # ptr bytes  next hotter page
                                #   4 bytes  this page's depth in the list
                                #   4 bytes  high-water mark
                                description += 'AUTORELEASE! for pthread_t %#x' % (thread)
                            else:
                                description += 'malloc(%u)' % (malloc_size)
                        else:
                            description += 'malloc(%u)' % (malloc_size)
                    else:
                        derefed_dynamic_value = dynamic_value.deref
                        if derefed_dynamic_value:                        
                            derefed_dynamic_type = derefed_dynamic_value.type
                            derefed_dynamic_type_size = derefed_dynamic_type.size
                            derefed_dynamic_type_name = derefed_dynamic_type.name
                            description += derefed_dynamic_type_name
                            if offset < derefed_dynamic_type_size:
                                member_list = list();
                                get_member_types_for_offset (derefed_dynamic_type, offset, member_list)
                                if member_list:
                                    member_path = ''
                                    for member in member_list:
                                        member_name = member.name
                                        if member_name: 
                                            if member_path:
                                                member_path += '.'
                                            member_path += member_name
                                    if member_path:
                                        if options.ivar_regex_blacklist:
                                            for ivar_regex in options.ivar_regex_blacklist:
                                                if ivar_regex.match(member_path):
                                                    print_entry = False
                                        description += '.%s' % (member_path)
                            else:
                                description += '%u bytes after %s' % (offset - derefed_dynamic_type_size, derefed_dynamic_type_name)
                        else:
                            # strip the "*" from the end of the name since we were unable to dereference this
                            description += dynamic_value.type.name[0:-1]
                if print_entry:
                    match_idx += 1
                    result_output = ''
                    if description:
                        result_output += description
                        if options.print_type and derefed_dynamic_value:
                            result_output += '%s' % (derefed_dynamic_value)
                        if options.print_object_description and dynamic_value:
                            desc = dynamic_value.GetObjectDescription()
                            if desc:
                                result_output += '\n%s' % (desc)
                    if result_output:
                        result.AppendMessage(result_output)
                    if options.memory:
                        cmd_result = lldb.SBCommandReturnObject()
                        memory_command = "memory read -f %s 0x%x 0x%x" % (options.format, malloc_addr, malloc_addr + malloc_size)
                        lldb.debugger.GetCommandInterpreter().HandleCommand(memory_command, cmd_result)
                        result.AppendMessage(cmd_result.GetOutput())
                    if options.stack_history:
                        dump_stack_history_entries(result, malloc_addr, 1)
                    elif options.stack:
                        dump_stack_history_entries(result, malloc_addr, 0)
            return i
        elif print_no_matches:
            result.AppendMessage('no matches found for %s' % (arg_str_description))
    else:
        result.AppendMessage(str(expr_sbvalue.error))
    return 0
    
def heap_search(result, options, arg_str):
    dylid_load_err = load_dylib()
    if dylid_load_err:
        result.AppendMessage(dylid_load_err)
        return
    expr = None
    print_no_matches = True
    arg_str_description = arg_str
    if options.type == 'pointer':
        expr = 'find_pointer_in_heap((void *)%s, (int)%u)' % (arg_str, options.check_vm_regions)
        arg_str_description = 'malloc block containing pointer %s' % arg_str
        if options.format == None: 
            options.format = "A" # 'A' is "address" format
    elif options.type == 'isa':
        expr = 'find_objc_objects_in_memory ((void *)%s, (int)%u)' % (arg_str, options.check_vm_regions)
        #result.AppendMessage ('expr -u0 -- %s' % expr) # REMOVE THIS LINE
        arg_str_description = 'objective C classes with isa %s' % arg_str
        options.offset = 0
        if options.format == None: 
            options.format = "A" # 'A' is "address" format
    elif options.type == 'cstr':
        expr = 'find_cstring_in_heap("%s", (int)%u)' % (arg_str, options.check_vm_regions)
        arg_str_description = 'malloc block containing "%s"' % arg_str
    elif options.type == 'addr':
        expr = 'find_block_for_address((void *)%s, (int)%u)' % (arg_str, options.check_vm_regions)
        arg_str_description = 'malloc block for %s' % arg_str
    elif options.type == 'all':
        expr = 'get_heap_info(1)'
        arg_str_description = None
        print_no_matches = False
    else:
        result.AppendMessage('error: invalid type "%s"\nvalid values are "pointer", "cstr"' % options.type)
        return
    if options.format == None: 
        options.format = "Y" # 'Y' is "bytes with ASCII" format
    
    display_match_results (result, options, arg_str_description, lldb.frame.EvaluateExpression (expr))
    
def ptr_refs(debugger, command, result, dict):
    command_args = shlex.split(command)
    usage = "usage: %prog [options] <EXPR> [EXPR ...]"
    description='''Searches the heap for pointer references on darwin user space programs. 
    
    Any matches that were found will dump the malloc blocks that contain the pointers 
    and might be able to print what kind of objects the pointers are contained in using 
    dynamic type information in the program.'''
    parser = optparse.OptionParser(description=description, prog='ptr_refs',usage=usage)
    add_common_options(parser)
    try:
        (options, args) = parser.parse_args(command_args)
    except:
        return

    options.type = 'pointer'
    
    if args:
        
        for data in args:
            heap_search (result, options, data)
    else:
        resultresult.AppendMessage('error: no pointer arguments were given')

def cstr_refs(debugger, command, result, dict):
    command_args = shlex.split(command)
    usage = "usage: %prog [options] <CSTR> [CSTR ...]"
    description='''Searches the heap for C string references on darwin user space programs. 
    
    Any matches that were found will dump the malloc blocks that contain the C strings 
    and might be able to print what kind of objects the pointers are contained in using 
    dynamic type information in the program.'''
    parser = optparse.OptionParser(description=description, prog='cstr_refs',usage=usage)
    add_common_options(parser)
    try:
        (options, args) = parser.parse_args(command_args)
    except:
        return

    options.type = 'cstr'

    if args:

        for data in args:
            heap_search (result, options, data)
    else:
        result.AppendMessage('error: no c string arguments were given to search for');

def malloc_info(debugger, command, result, dict):
    command_args = shlex.split(command)
    usage = "usage: %prog [options] <EXPR> [EXPR ...]"
    description='''Searches the heap a malloc block that contains the addresses specified as arguments. 

    Any matches that were found will dump the malloc blocks that match or contain
    the specified address. The matching blocks might be able to show what kind 
    of objects they are using dynamic type information in the program.'''
    parser = optparse.OptionParser(description=description, prog='cstr_refs',usage=usage)
    add_common_options(parser)
    try:
        (options, args) = parser.parse_args(command_args)
    except:
        return
    options.type = 'addr'
    if args:
        for data in args:
            heap_search (result, options, data)
    else:
        result.AppendMessage('error: no c string arguments were given to search for')

def heap(debugger, command, result, dict):
    command_args = shlex.split(command)
    usage = "usage: %prog [options] <EXPR> [EXPR ...]"
    description='''Traverse all allocations on the heap and report statistics.

    If programs set the MallocStackLogging=1 in the environment, then stack
    history is available for any allocations. '''
    parser = optparse.OptionParser(description=description, prog='cstr_refs',usage=usage)
    add_common_options(parser)
    try:
        (options, args) = parser.parse_args(command_args)
    except:
        return
    options.type = 'all'
    if args:
        result.AppendMessage('error: heap command takes no arguments, only options')
    else:
        heap_search (result, options, None)

def stack_ptr_refs(debugger, command, result, dict):
    command_args = shlex.split(command)
    usage = "usage: %prog [options] <EXPR> [EXPR ...]"
    description='''Searches thread stack contents for pointer values in darwin user space programs.'''
    parser = optparse.OptionParser(description=description, prog='section_ptr_refs',usage=usage)
    add_common_options(parser)
    try:
        (options, args) = parser.parse_args(command_args)
    except:
        return

    options.type = 'pointer'
    
    stack_threads = list()
    stack_bases = list()
    stack_sizes = list()
    for thread in lldb.process:
        min_sp = thread.frame[0].sp
        max_sp = min_sp
        for frame in thread.frames:
            sp = frame.sp
            if sp < min_sp: min_sp = sp
            if sp > max_sp: max_sp = sp
        result.AppendMessage ('%s stack [%#x - %#x)' % (thread, min_sp, max_sp))
        if min_sp < max_sp:
            stack_threads.append (thread)
            stack_bases.append (min_sp)
            stack_sizes.append (max_sp-min_sp)
        
    if stack_bases:
        dylid_load_err = load_dylib()
        if dylid_load_err:
            result.AppendMessage(dylid_load_err)
            return
        for expr_str in args:
            for (idx, stack_base) in enumerate(stack_bases):
                stack_size = stack_sizes[idx]
                expr = 'find_pointer_in_memory(0x%xllu, %ullu, (void *)%s)' % (stack_base, stack_size, expr_str)
                arg_str_description = 'thead %s stack containing "%s"' % (stack_threads[idx], expr_str)
                num_matches = display_match_results (result, options, arg_str_description, lldb.frame.EvaluateExpression (expr), False)
                if num_matches:
                    if num_matches < options.max_matches:
                        options.max_matches = options.max_matches - num_matches
                    else:
                        options.max_matches = 0
                if options.max_matches == 0:
                    return
    else:
        result.AppendMessage('error: no thread stacks were found that match any of %s' % (', '.join(options.section_names)))

def section_ptr_refs(debugger, command, result, dict):
    command_args = shlex.split(command)
    usage = "usage: %prog [options] <EXPR> [EXPR ...]"
    description='''Searches section contents for pointer values in darwin user space programs.'''
    parser = optparse.OptionParser(description=description, prog='section_ptr_refs',usage=usage)
    add_common_options(parser)
    parser.add_option('--section', action='append', type='string', dest='section_names', help='section name to search', default=list())
    try:
        (options, args) = parser.parse_args(command_args)
    except:
        return

    options.type = 'pointer'

    sections = list()
    section_modules = list()
    if not options.section_names:
        result.AppendMessage('error: at least one section must be specified with the --section option')
        return

    for module in lldb.target.modules:
        for section_name in options.section_names:
            section = module.section[section_name]
            if section:
                sections.append (section)
                section_modules.append (module)
    if sections:
        dylid_load_err = load_dylib()
        if dylid_load_err:
            result.AppendMessage(dylid_load_err)
            return
        for expr_str in args:
            for (idx, section) in enumerate(sections):
                expr = 'find_pointer_in_memory(0x%xllu, %ullu, (void *)%s)' % (section.addr.load_addr, section.size, expr_str)
                arg_str_description = 'section %s.%s containing "%s"' % (section_modules[idx].file.fullpath, section.name, expr_str)
                num_matches = display_match_results (result, options, arg_str_description, lldb.frame.EvaluateExpression (expr), False)
                if num_matches:
                    if num_matches < options.max_matches:
                        options.max_matches = options.max_matches - num_matches
                    else:
                        options.max_matches = 0
                if options.max_matches == 0:
                    return
    else:
        result.AppendMessage('error: no sections were found that match any of %s' % (', '.join(options.section_names)))

def objc_refs(debugger, command, result, dict):
    command_args = shlex.split(command)
    usage = "usage: %prog [options] <EXPR> [EXPR ...]"
    description='''Find all heap allocations given one or more objective C class names.'''
    parser = optparse.OptionParser(description=description, prog='object_refs',usage=usage)
    add_common_options(parser)
    try:
        (options, args) = parser.parse_args(command_args)
    except:
        return

    dylid_load_err = load_dylib()
    if dylid_load_err:
        result.AppendMessage(dylid_load_err)
    else:
        if args:
            for class_name in args:
                addr_expr_str = "(void *)[%s class]" % class_name
                expr_sbvalue = lldb.frame.EvaluateExpression (addr_expr_str)
                if expr_sbvalue.error.Success():
                    isa = expr_sbvalue.unsigned
                    if isa:
                        options.type = 'isa'
                        result.AppendMessage('Searching for all instances of classes or subclasses of %s (isa=0x%x)' % (class_name, isa))
                        heap_search (result, options, '0x%x' % isa)
                    else:
                        result.AppendMessage('error: Can\'t find isa for an ObjC class named "%s"' % (class_name))
                else:
                    result.AppendMessage('error: expression error for "%s": %s' % (addr_expr_str, expr_sbvalue.error))
        else:
            # Find all objective C objects by not specifying an isa
            options.type = 'isa'
            heap_search (result, options, '0x0')

if __name__ == '__main__':
    lldb.debugger = lldb.SBDebugger.Create()

# This initializer is being run from LLDB in the embedded command interpreter
# Add any commands contained in this module to LLDB
lldb.debugger.HandleCommand('command script add -f lldb.macosx.heap.ptr_refs ptr_refs')
lldb.debugger.HandleCommand('command script add -f lldb.macosx.heap.cstr_refs cstr_refs')
lldb.debugger.HandleCommand('command script add -f lldb.macosx.heap.malloc_info malloc_info')
lldb.debugger.HandleCommand('command script add -f lldb.macosx.heap.heap heap')
lldb.debugger.HandleCommand('command script add -f lldb.macosx.heap.section_ptr_refs section_ptr_refs')
lldb.debugger.HandleCommand('command script add -f lldb.macosx.heap.stack_ptr_refs stack_ptr_refs')
lldb.debugger.HandleCommand('command script add -f lldb.macosx.heap.objc_refs objc_refs')
print '"ptr_refs", "cstr_refs", "malloc_info", "heap", "section_ptr_refs" and "stack_ptr_refs" commands have been installed, use the "--help" options on these commands for detailed help.'




