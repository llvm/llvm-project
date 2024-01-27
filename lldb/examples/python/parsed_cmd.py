"""
This module implements a couple of utility classes to make writing
lldb parsed commands more Pythonic.
The way to use it is to make a class for your command that inherits from ParsedCommandBase.
That will make an LLDBOVParser which you will use for your
option definition, and to fetch option values for the current invocation
of your command.  Access to the OV parser is through:

ParsedCommandBase.get_parser()

Next, implement setup_command_definition() in your new command class, and call:

  self.get_parser().add_option()

to add all your options.  The order doesn't matter for options, lldb will sort them
alphabetically for you when it prints help.

Similarly you can define the arguments with:

  self.get_parser().add_argument()

At present, lldb doesn't do as much work as it should verifying arguments, it
only checks that commands that take no arguments don't get passed arguments.

Then implement the execute function for your command as:

    def __call__(self, debugger, args_list, exe_ctx, result):

The arguments will be a list of strings.  

You can access the option values using the 'varname' string you passed in when defining the option.

If you need to know whether a given option was set by the user or not, you can retrieve 
the option definition array with:

  self.get_options_definition()

then look up your element by the 'varname' field, and check the "_value_set" element.
FIXME: I should add a convenience method to do this.

There are example commands in the lldb testsuite at:

llvm-project/lldb/test/API/commands/command/script/add/test_commands.py
"""
import inspect
import lldb
import sys
from abc import abstractmethod

class LLDBOVParser:
    def __init__(self):
        # This is a dictionary of dictionaries.  The key is the long option
        # name, and the value is the rest of the definition.
        self.options_dict = {}
        self.args_array = []

    # Some methods to translate common value types.  Should return a
    # tuple of the value and an error value (True => error) if the
    # type can't be converted.
    # FIXME: Need a way to push the conversion error string back to lldb.
    @staticmethod
    def to_bool(in_value):
        error = True
        value = False
        print(f"TYPE: {type(in_value)}")
        if type(in_value) != str or len(in_value) == 0:
            return (value, error)

        low_in = in_value.lower()
        if low_in == "y" or low_in == "yes" or low_in == "t" or low_in == "true" or low_in == "1":
            value = True
            error = False
            
        if not value and low_in == "n" or low_in == "no" or low_in == "f" or low_in == "false" or low_in == "0":
            value = False
            error = False

        return (value, error)

    @staticmethod
    def to_int(in_value):
        #FIXME: Not doing errors yet...
        return (int(in_value), False)

    @staticmethod
    def to_unsigned(in_value):
        # FIXME: find an unsigned converter...
        # And handle errors.
        return (int(in_value), False)

    translators = {
        lldb.eArgTypeBoolean : to_bool,
        lldb.eArgTypeBreakpointID : to_unsigned,
        lldb.eArgTypeByteSize : to_unsigned,
        lldb.eArgTypeCount : to_unsigned,
        lldb.eArgTypeFrameIndex : to_unsigned,
        lldb.eArgTypeIndex : to_unsigned,
        lldb.eArgTypeLineNum : to_unsigned,
        lldb.eArgTypeNumLines : to_unsigned,
        lldb.eArgTypeNumberPerLine : to_unsigned,
        lldb.eArgTypeOffset : to_int,
        lldb.eArgTypeThreadIndex : to_unsigned,
        lldb.eArgTypeUnsignedInteger : to_unsigned,
        lldb.eArgTypeWatchpointID : to_unsigned,
        lldb.eArgTypeColumnNum : to_unsigned,
        lldb.eArgTypeRecognizerID : to_unsigned,
        lldb.eArgTypeTargetID : to_unsigned,
        lldb.eArgTypeStopHookID : to_unsigned
    }

    @classmethod
    def translate_value(cls, value_type, value):
        try:
            return cls.translators[value_type](value)
        except KeyError:
            # If we don't have a translator, return the string value.
            return (value, False)

    # FIXME: would this be better done on the C++ side?
    # The common completers are missing some useful ones.
    # For instance there really should be a common Type completer
    # And an "lldb command name" completer.
    completion_table = {
        lldb.eArgTypeAddressOrExpression : lldb.eVariablePathCompletion,
        lldb.eArgTypeArchitecture : lldb.eArchitectureCompletion,
        lldb.eArgTypeBreakpointID : lldb.eBreakpointCompletion,
        lldb.eArgTypeBreakpointIDRange : lldb.eBreakpointCompletion,
        lldb.eArgTypeBreakpointName : lldb.eBreakpointNameCompletion,
        lldb.eArgTypeClassName : lldb.eSymbolCompletion,
        lldb.eArgTypeDirectoryName : lldb.eDiskDirectoryCompletion,
        lldb.eArgTypeExpression : lldb.eVariablePathCompletion,
        lldb.eArgTypeExpressionPath : lldb.eVariablePathCompletion,
        lldb.eArgTypeFilename : lldb.eDiskFileCompletion,
        lldb.eArgTypeFrameIndex : lldb.eFrameIndexCompletion,
        lldb.eArgTypeFunctionName : lldb.eSymbolCompletion,
        lldb.eArgTypeFunctionOrSymbol : lldb.eSymbolCompletion,
        lldb.eArgTypeLanguage : lldb.eTypeLanguageCompletion,
        lldb.eArgTypePath : lldb.eDiskFileCompletion,
        lldb.eArgTypePid : lldb.eProcessIDCompletion,
        lldb.eArgTypeProcessName : lldb.eProcessNameCompletion,
        lldb.eArgTypeRegisterName : lldb.eRegisterCompletion,
        lldb.eArgTypeRunArgs : lldb.eDiskFileCompletion,
        lldb.eArgTypeShlibName : lldb.eModuleCompletion,
        lldb.eArgTypeSourceFile : lldb.eSourceFileCompletion,
        lldb.eArgTypeSymbol : lldb.eSymbolCompletion,
        lldb.eArgTypeThreadIndex : lldb.eThreadIndexCompletion,
        lldb.eArgTypeVarName : lldb.eVariablePathCompletion,
        lldb.eArgTypePlatform : lldb.ePlatformPluginCompletion,
        lldb.eArgTypeWatchpointID : lldb.eWatchpointIDCompletion,
        lldb.eArgTypeWatchpointIDRange : lldb.eWatchpointIDCompletion,
        lldb.eArgTypeModuleUUID : lldb.eModuleUUIDCompletion,
        lldb.eArgTypeStopHookID : lldb.eStopHookIDCompletion
    }

    @classmethod
    def determine_completion(cls, arg_type):
        try:
            return cls.completion_table[arg_type]
        except KeyError:
            return lldb.eNoCompletion

    def get_option_element(self, long_name):
        # Fixme: Is it worth making a long_option dict holding the rest of
        # the options dict so this lookup is faster?
        return self.options_dict.get(long_name, None)
            
    def option_parsing_started(self):
        # This makes the ivars for all the varnames in the array and gives them
        # their default values.
        for key, elem in self.options_dict.items():
            #breakpoint()
            elem['_value_set'] = False
            try:
                object.__setattr__(self, elem["varname"], elem["default"])
            except AttributeError:
            # It isn't an error not to have a target, you'll just have to set and
            # get this option value on your own.
                continue

    def set_enum_value(self, enum_values, input):
        candidates = []
        for candidate in enum_values:
            # The enum_values are a two element list of value & help string.
            value = candidate[0]
            if value.startswith(input):
                candidates.append(value)

        if len(candidates) == 1:
            return (candidates[0], False)
        else:
            return (input, True)
        
    def set_option_value(self, exe_ctx, opt_name, opt_value):
        elem = self.get_option_element(opt_name)
        if not elem:
            return False
        
        if "enum_values" in elem:
            (value, error) = self.set_enum_value(elem["enum_values"], opt_value)
        else:
            (value, error)  = __class__.translate_value(elem["value_type"], opt_value)

        if not error:
            object.__setattr__(self, elem["varname"], value)
            elem["_value_set"] = True
            return True
        return False

    def was_set(self, opt_name):
        elem = self.get_option_element(opt_name)
        if not elem:
            return False
        try:
            return elem["_value_set"]
        except AttributeError:
            return False

    def is_enum_opt(self, opt_name):
        elem = self.get_option_element(opt_name)
        if not elem:
            return False
        return "enum_values" in elem

    def add_option(self, short_option, long_option, usage, default,
                   varname = None, required=False, groups = None,
                   value_type=lldb.eArgTypeNone, completion_type=None,
                   enum_values=None):
        """
        short_option: one character, must be unique, not required
        long_option: no spaces, must be unique, required
        usage: a usage string for this option, will print in the command help
        default: the initial value for this option (if it has a value)
        varname: the name of the property that gives you access to the value for
                 this value.  Defaults to the long option if not provided.
        required: if true, this option must be provided or the command will error out
        groups: Which "option groups" does this option belong to
        value_type: one of the lldb.eArgType enum values.  Some of the common arg
                    types also have default completers, which will be applied automatically.
        completion_type: currently these are values form the lldb.CompletionType enum, I
                         haven't done custom completions yet.
        enum_values: An array of duples: ["element_name", "element_help"].  If provided,
                     only one of the enum elements is allowed.  The value will be the 
                     element_name for the chosen enum element as a string. 
        """
        if not varname:
            varname = long_option

        if not completion_type:
            completion_type = self.determine_completion(value_type)
            
        dict = {"short_option" : short_option,
                "required" : required,
                "usage" : usage,
                "value_type" : value_type,
                "completion_type" : completion_type,
                "varname" : varname,
                "default" : default}

        if enum_values:
            dict["enum_values"] = enum_values
        if groups:
            dict["groups"] = groups

        self.options_dict[long_option] = dict

    def make_argument_element(self, arg_type, repeat = "optional", groups = None):
        element = {"arg_type" : arg_type, "repeat" : repeat}
        if groups:
            element["groups"] = groups
        return element

    def add_argument_set(self, arguments):
        self.args_array.append(arguments)

class ParsedCommandBase:
    def __init__(self, debugger, unused):
        self.debugger = debugger
        self.ov_parser = LLDBOVParser()
        self.setup_command_definition()
        
    def get_parser(self):
        return self.ov_parser

    def get_options_definition(self):
        return self.get_parser().options_dict

    def get_flags(self):
        return 0

    def get_args_definition(self):
        return self.get_parser().args_array

    def option_parsing_started(self):
        self.get_parser().option_parsing_started()

    def set_option_value(self, exe_ctx, opt_name, opt_value):
        return self.get_parser().set_option_value(exe_ctx, opt_name, opt_value)

    # These are the two "pure virtual" methods:
    @abstractmethod
    def __call__(self, debugger, args_array, exe_ctx, result):
        raise NotImplementedError()

    @abstractmethod
    def setup_command_definition(self):
        raise NotImplementedError()

    @staticmethod
    def do_register_cmd(cls, debugger, module_name):
        # Add any commands contained in this module to LLDB
        command = "command script add -o -p -c %s.%s %s" % (
            module_name,
            cls.__name__,
            cls.program,
        )
        debugger.HandleCommand(command)
        print(
            'The "{0}" command has been installed, type "help {0}"'
            'for detailed help.'.format(cls.program)
        )
