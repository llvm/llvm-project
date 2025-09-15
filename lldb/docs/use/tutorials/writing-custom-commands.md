# Writing Custom Commands

### Create a new command using a Python function

Python functions can be used to create new LLDB command interpreter commands,
which will work like all the natively defined lldb commands. This provides a
very flexible and easy way to extend LLDB to meet your debugging requirements.

To write a python function that implements a new LLDB command define the
function to take five arguments as follows:

```python3
def command_function(debugger, command, exe_ctx, result, internal_dict):
    # Your code goes here
```

The meaning of the arguments is given in the table below.

If you provide a Python docstring in your command function LLDB will use it
when providing "long help" for your command, as in:

```python3
def command_function(debugger, command, result, internal_dict):
    """This command takes a lot of options and does many fancy things"""
    # Your code goes here
```

though providing help can also be done programmatically (see below).

Prior to lldb 3.5.2 (April 2015), LLDB Python command definitions didn't take the SBExecutionContext
argument. So you may still see commands where the command definition is:

```python3
def command_function(debugger, command, result, internal_dict):
    # Your code goes here
```

Using this form is strongly discouraged because it can only operate on the "currently selected"
target, process, thread, frame.  The command will behave as expected when run
directly on the command line.  But if the command is used in a stop-hook, breakpoint
callback, etc. where the response to the callback determines whether we will select
this or that particular process/frame/thread, the global "currently selected"
entity is not necessarily the one the callback is meant to handle.  In that case, this
command definition form can't do the right thing.

| Argument | Type | Description |
|----------|------|-------------|
| `debugger` | `lldb.SBDebugger` | The current debugger object. |
| `command` | `python string` | A python string containing all arguments for your command. If you need to chop up the arguments try using the `shlex` module's `shlex.split(command)` to properly extract the arguments. |
| `exe_ctx` | `lldb.SBExecutionContext` | An execution context object carrying around information on the inferior process' context in which the command is expected to act *Optional since lldb 3.5.2, unavailable before* |
| `result` | `lldb.SBCommandReturnObject` | A return object which encapsulates success/failure information for the command and output text that needs to be printed as a result of the command. The plain Python "print" command also works but text won't go in the result by default (it is useful as a temporary logging facility). |
| `internal_dict` | `python dict object` | The dictionary for the current embedded script session which contains all variables and functions. |

### Create a new command using a Python class

Since lldb 3.7, Python commands can also be implemented by means of a class
which should implement the following interface:

```python3
class CommandObjectType:
    def __init__(self, debugger, internal_dict):
        # this call should initialize the command with respect to the command interpreter for the passed-in debugger

    def __call__(self, debugger, command, exe_ctx, result):
        # this is the actual bulk of the command, akin to Python command functions

    def get_short_help(self):
        # this call should return the short help text for this command[1]

    def get_long_help(self):
        # this call should return the long help text for this command[1]

    def get_flags(self):
        # this will be called when the command is added to the command interpreter,
        # and should return a flag field made from or-ing together the appropriate
        # elements of the lldb.CommandFlags enum to specify the requirements of this command.
        # The CommandInterpreter will make sure all these requirements are met, and will
        # return the standard lldb error if they are not.[1]

    def get_repeat_command(self, command):
        # The auto-repeat command is what will get executed when the user types just
        # a return at the next prompt after this command is run.  Even if your command
        # was run because it was specified as a repeat command, that invocation will still
        # get asked for IT'S repeat command, so you can chain a series of repeats, for instance
        # to implement a pager.

        # The command argument is the command that is about to be executed.

        # If this call returns None, then the ordinary repeat mechanism will be used
        # If this call returns an empty string, then auto-repeat is disabled
        # If this call returns any other string, that will be the repeat command [1]
```

[1] This method is optional.

As a convenience, you can treat the result object as a Python file object, and
say

```python3
print("my command does lots of cool stuff", file=result)
```

`SBCommandReturnObject` and `SBStream` both support this file-like behavior by
providing `write()` and `flush()` calls at the Python layer.

### Parsed Commands

The commands that are added using this class definition are what lldb calls
"raw" commands.  The command interpreter doesn't attempt to parse the command,
doesn't handle option values, neither generating help for them, or their
completion.  Raw commands are useful when the arguments passed to the command
are unstructured, and having to protect them against lldb command parsing would
be onerous.  For instance, "expr" is a raw command.

You can also add scripted commands that implement the "parsed command", where
the options and their types are specified, as well as the argument and argument
types.  These commands look and act like the majority of lldb commands, and you
can also add custom completions for the options and/or the arguments if you have
special needs.

The easiest way to do this is to derive your new command from the lldb.ParsedCommand
class.  That responds in the same way to the help & repeat command interfaces, and
provides some convenience methods, and most importantly an LLDBOptionValueParser,
accessed through lldb.ParsedCommand.get_parser().  The parser is used to set
your command definitions, and to retrieve option values in the `__call__` method.

To set up the command definition, implement the ParsedCommand abstract method:

```python3
def setup_command_definition(self):
```

This is called when your command is added to lldb.  In this method you add the
options and their types, the option help strings, etc. to the command using the API:

```python3
def add_option(self, short_option, long_option, help, default,
               dest = None, required=False, groups = None,
               value_type=lldb.eArgTypeNone, completion_type=None,
               enum_values=None):
    """
    short_option: one character, must be unique, not required
    long_option:  no spaces, must be unique, required
    help:         a usage string for this option, will print in the command help
    default:      the initial value for this option (if it has a value)
    dest:         the name of the property that gives you access to the value for
                  this value.  Defaults to the long option if not provided.
    required: if true, this option must be provided or the command will error out
    groups: Which "option groups" does this option belong to.  This can either be
            a simple list (e.g. [1, 3, 4, 5]) or you can specify ranges by sublists:
            so [1, [3,5]] is the same as [1, 3, 4, 5].
    value_type: one of the lldb.eArgType enum values.  Some of the common arg
                types also have default completers, which will be applied automatically.
    completion_type: currently these are values form the lldb.CompletionType enum.	If
                     you need custom completions, implement	handle_option_argument_completion.
    enum_values: An array of duples: ["element_name", "element_help"].  If provided,
                 only one of the enum elements is allowed.  The value will be the
                 element_name for the chosen enum element as a string.
    """
```

Similarly, you can add argument types to the command:

```python3
def make_argument_element(self, arg_type, repeat = "optional", groups = None):
    """
  	arg_type: The argument type, one of the	lldb.eArgType enum values.
  	repeat:	Choose from the	following options:
  	      	"plain"	- one value
  	      	"optional" - zero or more values
  	      	"plus" - one or	more values
  	groups:	As with	add_option.
    """
```

Then implement the body of the command by defining:

```python3
def __call__(self, debugger, args_array, exe_ctx, result):
    """This is the command callback.  The option values are
    provided by the 'dest' properties on the parser.

    args_array: This is the list of arguments provided.
    exe_ctx: Gives the SBExecutionContext on which the
             command should operate.
    result:  Any results of the command should be
             written into this SBCommandReturnObject.
    """
```

This differs from the "raw" command's `__call__` in that the arguments are already
parsed into the args_array, and the option values are set in the parser, and
can be accessed using their property name.  The LLDBOptionValueParser class has
a couple of other handy methods:

```python3
def was_set(self, long_option_name):
```

returns `True` if the option was specified on the command line.

```python
def dest_for_option(self, long_option_name):
"""
This will return the value of the dest variable you defined for opt_name.
Mostly useful for handle_completion where you get passed the long option.
"""
```

### Completion

lldb will handle completing your option names, and all your enum values
automatically.  If your option or argument types have associated built-in completers,
then lldb will also handle that completion for you.  But if you have a need for
custom completions, either in your arguments or option values, you can handle
completion by hand as well.  To handle completion of option value arguments,
your lldb.ParsedCommand subclass should implement:

```python3
def handle_option_argument_completion(self, long_option, cursor_pos):
"""
long_option: The long option name of the option whose value you are
             asked to complete.
cursor_pos: The cursor position in the value for that option - which
you can get from the option parser.
"""
```

And to handle the completion of arguments:

```python3
def handle_argument_completion(self, args, arg_pos, cursor_pos):
"""
args: A list of the arguments to the command
arg_pos: An index into the args list of the argument with the cursor
cursor_pos: The cursor position in the arg specified by arg_pos
"""
```

When either of these API's is called, the command line will have been parsed up to
the word containing the cursor, and any option values set in that part of the command
string are available from the option value parser.  That's useful for instance
if you have a --shared-library option that would constrain the completions for,
say, a symbol name option or argument.

The return value specifies what the completion options are.  You have four
choices:

- `True`: the completion was handled with no completions.

- `False`: the completion was not handled, forward it to the regular
completion machinery.

- A dictionary with the key: "completion": there is one candidate,
whose value is the value of the "completion" key.  Optionally you can pass a
"mode" key whose value is either "partial" or "complete".  Return partial if
the "completion" string is a prefix for all the completed value.

For instance, if the string you are completing is "Test" and the available completions are:
"Test1", "Test11" and "Test111", you should return the dictionary:

```python3
return {"completion": "Test1", "mode" : "partial"}
```

and then lldb will add the "1" at the cursor and advance it after the added string,
waiting for more completions.  But if "Test1" is the only completion, return:

```python3
{"completion": "Test1", "mode": "complete"}
```

and lldb will add "1 " at the cursor, indicating the command string is complete.

The default is "complete", you don't need to specify a "mode" in that case.

- A dictionary with the key: "values" whose value is a list of candidate completion
strings.  The command interpreter will present those strings as the available choices.
You can optionally include a "descriptions" key, whose value is a parallel array
of description strings, and the completion will show the description next to
each completion.

### Loading Commands

One other handy convenience when defining lldb command-line commands is the
command "command script import" which will import a module specified by file
path, so you don't have to change your PYTHONPATH for temporary scripts. It
also has another convenience that if your new script module has a function of
the form:

```python
def __lldb_init_module(debugger, internal_dict):
    # Command Initialization code goes here
```

where debugger and internal_dict are as above, that function will get run when
the module is loaded allowing you to add whatever commands you want into the
current debugger. Note that this function will only be run when using the LLDB
command `command script import`, it will not get run if anyone imports your
module from another module.

Another way to load custom commands in lldb is to use the
`@lldb.command(command_name=None, doc=None)` decorator.

```python3
@lldb.command()
def goodstuff(debugger, command, ctx, result, internal_dict):
    """command help string"""
    # Command Implementation code goes here
```

### Examples

Now we can create a module called ls.py in the file ~/ls.py that will implement
a function that can be used by LLDB's python command code:

```python3
#!/usr/bin/env python3

import lldb
import subprocess

def ls(debugger, command, result, internal_dict):
    output = subprocess.check_output(["/bin/ls"] + command.split(), text=True)
    print(output, file=result)

# And the initialization code to add your commands
def __lldb_init_module(debugger, internal_dict):
    debugger.HandleCommand('command script add -f ls.ls ls')
    print('The "ls" python command has been installed and is ready for use.')
```

Now we can load the module into LLDB and use it

```shell
$ lldb
(lldb) command script import ~/ls.py
The "ls" python command has been installed and is ready for use.
(lldb) ls -l /tmp/
total 365848
-rw-------   1 someuser  wheel         7331 Jan 19 15:37 crash.log
```

You can also make "container" commands to organize the commands you are adding to
lldb.  Most of the lldb built-in commands structure themselves this way, and using
a tree structure has the benefit of leaving the one-word command space free for user
aliases.  It can also make it easier to find commands if you are adding more than
a few of them.  Here's a trivial example of adding two "utility" commands into a
"my-utilities" container:

```python3
#!/usr/bin/env python

import lldb

def first_utility(debugger, command, result, internal_dict):
    print("I am the first utility")

def second_utility(debugger, command, result, internal_dict):
    print("I am the second utility")

# And the initialization code to add your commands
def __lldb_init_module(debugger, internal_dict):
    debugger.HandleCommand('command container add -h "A container for my utilities" my-utilities')
    debugger.HandleCommand('command script add -f my_utilities.first_utility -h "My first utility" my-utilities first')
    debugger.HandleCommand('command script add -f my_utilities.second_utility -h "My second utility" my-utilities second')
    print('The "my-utilities" python command has been installed and its subcommands are ready for use.')
```

Then your new commands are available under the my-utilities node:

```
(lldb) help my-utilities
A container for my utilities

Syntax: my-utilities

The following subcommands are supported:

    first  -- My first utility  Expects 'raw' input (see 'help raw-input'.)
    second -- My second utility  Expects 'raw' input (see 'help raw-input'.)

For more help on any particular subcommand, type 'help <command> <subcommand>'.
(lldb) my-utilities first
I am the first utility
```

A more interesting [template](https://github.com/llvm/llvm-project/blob/main/lldb/examples/python/cmdtemplate.py)
has been created in the source repository that can help you to create lldb command quickly.

A commonly required facility is being able to create a command that does some
token substitution, and then runs a different debugger command (usually, it
po'es the result of an expression evaluated on its argument). For instance,
given the following program:

```objc
#import <Foundation/Foundation.h>
NSString*
ModifyString(NSString* src)
{
	return [src stringByAppendingString:@"foobar"];
}

int main()
{
	NSString* aString = @"Hello world";
	NSString* anotherString = @"Let's be friends";
	return 1;
}
```

you may want a `pofoo` X command, that equates po [ModifyString(X)
capitalizedString]. The following debugger interaction shows how to achieve
that goal:

```python3
(lldb) script
Python Interactive Interpreter. To exit, type 'quit()', 'exit()' or Ctrl-D.
>>> def pofoo_funct(debugger, command, result, internal_dict):
...	cmd = "po [ModifyString(" + command + ") capitalizedString]"
...	debugger.HandleCommand(cmd)
...
>>> ^D
(lldb) command script add pofoo -f pofoo_funct
(lldb) pofoo aString
$1 = 0x000000010010aa00 Hello Worldfoobar
(lldb) pofoo anotherString
$2 = 0x000000010010aba0 Let's Be Friendsfoobar
```