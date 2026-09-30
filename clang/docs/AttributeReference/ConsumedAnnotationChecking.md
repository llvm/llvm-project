## Consumed Annotation Checking

Clang supports additional attributes for checking basic resource management
properties, specifically for unique objects that have a single owning reference.
The following attributes are currently supported, although **the implementation
for these annotations is currently in development and are subject to change.**

### callable_when

{clang-attr-syntaxes}`CallableWhenDocs`

Use `__attribute__((callable_when(...)))` to indicate what states a method
may be called in. Valid states are unconsumed, consumed, or unknown. Each
argument to this attribute must be a quoted string. E.g.:

`__attribute__((callable_when("unconsumed", "unknown")))`


### consumable

{clang-attr-syntaxes}`ConsumableDocs`

Each `class` that uses any of the typestate annotations must first be marked
using the `consumable` attribute. Failure to do so will result in a warning.

This attribute accepts a single parameter that must be one of the following:
`unknown`, `consumed`, or `unconsumed`.


### param_typestate

{clang-attr-syntaxes}`ParamTypestateDocs`

This attribute specifies expectations about function parameters. Calls to an
function with annotated parameters will issue a warning if the corresponding
argument isn't in the expected state. The attribute is also used to set the
initial state of the parameter when analyzing the function's body.


### return_typestate

{clang-attr-syntaxes}`ReturnTypestateDocs`

The `return_typestate` attribute can be applied to functions or parameters.
When applied to a function the attribute specifies the state of the returned
value. The function's body is checked to ensure that it always returns a value
in the specified state. On the caller side, values returned by the annotated
function are initialized to the given state.

When applied to a function parameter it modifies the state of an argument after
a call to the function returns. The function's body is checked to ensure that
the parameter is in the expected state before returning.


### set_typestate

{clang-attr-syntaxes}`SetTypestateDocs`

Annotate methods that transition an object into a new state with
`__attribute__((set_typestate(new_state)))`. The new state must be
unconsumed, consumed, or unknown.


### test_typestate

{clang-attr-syntaxes}`TestTypestateDocs`

Use `__attribute__((test_typestate(tested_state)))` to indicate that a method
returns true if the object is in the specified state..


