# Custom Breakpoint Resolvers

Another use of the Python API's in lldb is to create a custom breakpoint
resolver.

It allows you to provide the algorithm which will be used in the breakpoint's
search of the space of the code in a given Target to determine where to set the
breakpoint locations - the actual places where the breakpoint will trigger. To
understand how this works you need to know a little about how lldb handles
breakpoints.

In lldb, a breakpoint is composed of three parts:
1. the Searcher
2. the Resolver,
3. the Stop Options.

The Searcher and Resolver cooperate to determine how breakpoint locations are
set and differ between each breakpoint type. Stop options determine what
happens when a location triggers and includes the commands, conditions, ignore
counts, etc. Stop options are common between all breakpoint types, so for our
purposes only the Searcher and Resolver are relevant.

### Breakpoint Searcher

The Searcher's job is to traverse in a structured way the code in the current
target. It proceeds from the Target, to search all the Modules in the Target,
in each Module it can recurse into the Compile Units in that module, and within
each Compile Unit it can recurse over the Functions it contains.

The Searcher can be provided with a SearchFilter that it will use to restrict
this search. For instance, if the SearchFilter specifies a list of Modules, the
Searcher will not recurse into Modules that aren't on the list. When you pass
the -s modulename flag to break set you are creating a Module-based search
filter. When you pass -f filename.c to break set -n you are creating a file
based search filter. If neither of these is specified, the breakpoint will have
a no-op search filter, so all parts of the program are searched and all
locations accepted.

### Breakpoint Resolver

The Resolver has two functions:

The most important one is the callback it provides. This will get called at the
appropriate time in the course of the search. The callback is where the job of
adding locations to the breakpoint gets done.

The other function is specifying to the Searcher at what depth in the above
described recursion it wants to be called. Setting a search depth also provides
a stop for the recursion. For instance, if you request a Module depth search,
then the callback will be called for each Module as it gets added to the
Target, but the searcher will not recurse into the Compile Units in the module.

One other slight subtlety is that the depth at which you get called back is not
necessarily the depth at which the SearchFilter is specified. For instance,
if you are doing symbol searches, it is convenient to use the Module depth for
the search, since symbols are stored in the module. But the SearchFilter might
specify some subset of CompileUnits, so not all the symbols you might find in
each module will pass the search. You don't need to handle this situation
yourself, since SBBreakpoint::AddLocation will only add locations that pass the
Search Filter. This API returns an SBError to inform you whether your location
was added.

When the breakpoint is originally created, its Searcher will process all the
currently loaded modules. The Searcher will also visit any new modules as they
are added to the target. This happens, for instance, when a new shared library
gets added to the target in the course of running, or on rerunning if any of
the currently loaded modules have been changed. Note, in the latter case, all
the locations set in the old module will get deleted and you will be asked to
recreate them in the new version of the module when your callback gets called
with that module. For this reason, you shouldn't try to manage the locations
you add to the breakpoint yourself. Note that the Breakpoint takes care of
deduplicating equal addresses in AddLocation, so you shouldn't need to worry
about that anyway.

### Scripted Breakpoint Resolver

At present, when adding a ScriptedBreakpoint type, you can only provide a
custom Resolver, not a custom SearchFilter.

The custom Resolver is provided as a Python class with the following methods:

| Name | Arguments | Description |
|------|-----------|-------------|
| `__init__` | `bkpt`: `lldb.SBBreakpoint` `extra_args`: `lldb.SBStructuredData` | This is the constructor for the new Resolver. `bkpt` is the breakpoint owning this Resolver. `extra_args` is an `SBStructuredData` object that the user can pass in when creating instances of this breakpoint. It is not required, but is quite handy. For instance if you were implementing a breakpoint on some symbol name, you could write a generic symbol name based Resolver, and then allow the user to pass in the particular symbol in the extra_args |
| `__callback__` | `sym_ctx`: `lldb.SBSymbolContext` | This is the Resolver callback. The `sym_ctx` argument will be filled with the current stage of the search. For instance, if you asked for a search depth of lldb.eSearchDepthCompUnit, then the target, module and compile_unit fields of the sym_ctx will be filled. The callback should look just in the context passed in `sym_ctx` for new locations. If the callback finds an address of interest, it can add it to the breakpoint with the `SBBreakpoint.AddLocation` method, using the breakpoint passed in to the `__init__` method. |
| `__get_depth__` | `None` | Specify the depth at which you wish your callback to get called. The currently supported options are: `lldb.eSearchDepthModule` `lldb.eSearchDepthCompUnit` `lldb.eSearchDepthFunction` For instance, if you are looking up symbols, which are stored at the Module level, you will want to get called back module by module. So you would want to return `lldb.eSearchDepthModule`. This method is optional. If not provided the search will be done at Module depth. |
| `get_short_help` | `None` | This is an optional method. If provided, the returned string will be printed at the beginning of the description for this breakpoint. |

To define a new breakpoint command defined by this class from the lldb command
line, use the command:

```
(lldb) breakpoint set -P MyModule.MyResolverClass
```

You can also populate the extra_args SBStructuredData with a dictionary of
key/value pairs with:

```
(lldb) breakpoint set -P MyModule.MyResolverClass -k key_1 -v value_1 -k key_2 -v value_2
```

Although you can't write a scripted SearchFilter, both the command line and the
SB API's for adding a scripted resolver allow you to specify a SearchFilter
restricted to certain modules or certain compile units. When using the command
line to create the resolver, you can specify a Module specific SearchFilter by
passing the -s ModuleName option - which can be specified multiple times. You
can also specify a SearchFilter restricted to certain compile units by passing
in the -f CompUnitName option. This can also be specified more than once. And
you can mix the two to specify "this comp unit in this module". So, for
instance,

```
(lldb) breakpoint set -P MyModule.MyResolverClass -s a.out
```

will use your resolver, but will only recurse into or accept new locations in
the module a.out.

Another option for creating scripted breakpoints is to use the
SBTarget.BreakpointCreateFromScript API. This one has the advantage that you
can pass in an arbitrary SBStructuredData object, so you can create more
complex parametrizations. SBStructuredData has a handy SetFromJSON method which
you can use for this purpose. Your __init__ function gets passed this
SBStructuredData object. This API also allows you to directly provide the list
of Modules and the list of CompileUnits that will make up the SearchFilter. If
you pass in empty lists, the breakpoint will use the default "search
everywhere,accept everything" filter.

### Providing Facade Locations:

The breakpoint resolver interface also allows you to present a separate set
of locations for the breakpoint than the ones that actually implement the
breakpoint in the target.

An example use case for this is if you are providing a debugging interface for a
library that implements an interpreter for a language lldb can't debug.  But
while debugging that library at the level of the implementation language (e.g. C/C++, etc)
you would like to offer the ability to "stop when a line in a source language
file is executed".

You can do this if you know where new lines of code are dispatched in the
interpreter.  You would set a breakpoint there, and then look at the state
when that breakpoint is hit to see if it is dispatching the source file and
line that were requested, and stop appropriately.

Facade breakpoint locations are intended to make a more natural presentation
of that sort of feature.  The idea is that you would make a custom breakpoint
resolver that sets actual locations in the places of interest in the interpreter.

Then your resolver would add "facade locations" that represent the places in the
interpreted code that you want the breakpoint to stop at, using SBBreakpoint::AddFacadeLocation.
When lldb describes the breakpoint, it will only show the Facade locations.
Since facade breakpoint location's description is customizable, you can make these
locations more descriptive.  And when the "real" location is hit, lldb will call the
"was_hit" method of your resolver.  That will return the facade location you
consider to have been hit this time around, or if you return None, the breakpoint
will be considered not to have been hit.

Note, this feature is also useful if you don't intend to present facade
locations since it essentially provides a scripted breakpoint condition.  Every
time one of the locations in your breakpoint is hit, you can run the code in
your "was_hit" to determine whether to consider the breakpoint hit or not, and
return the location you were passed in if you want it to be a hit, and None if not.

The Facade location adds these optional affordances to the Resolver class:

| Name  | Arguments | Description|
|-------|-----------|------------|
|`was_hit`| `frame`:`lldb.SBFrame` `bp_loc`:`lldb.SBBreakpointLocation` | This will get called when one of the "real" locations set by your resolver is hit.  `frame` is the stack frame that hit this location.  `bp_loc` is the real location that was hit.  Return either the facade location that you want to consider hit on this stop, or None if you don't consider any of your facade locations to have been hit. |
| `get_location_description` | `bp_loc`:`lldb.SBBreakpointLocation` `desc_level`:`lldb.DescriptionLevel` `bp_loc` is the facade location to describe.| Use this to provide a helpful description of each facade location.  ``desc_level`` is the level of description requested.  The Brief description is printed when the location is hit.  Full is printed for `break list` and Verbose for `break list -v`.|

