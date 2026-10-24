## AArch64 SME Attributes

Clang supports a number of AArch64-specific attributes to manage state
added by the Scalable Matrix Extension (SME). This state includes the
runtime mode that the processor is in (e.g. non-streaming or streaming)
as well as the state of the `ZA` Matrix Storage.

The attributes come in the form of type- and declaration attributes:

- The SME declaration attributes can appear anywhere that a standard
  `[[...]]` declaration attribute can appear.
- The SME type attributes apply only to prototyped functions and can appear
  anywhere that a standard `[[...]]` type attribute can appear. The SME
  type attributes do not apply to functions having a K&R-style
  unprototyped function type.

See [Arm C Language Extensions](https://github.com/ARM-software/acle)
for more details about the features related to the SME extension.

See [Procedure Call Standard for the Arm® 64-bit Architecture (AArch64)](https://github.com/ARM-software/abi-aa) for more details about
streaming-interface functions and shared/private-ZA interface functions.

### __arm_agnostic

{clang-attr-syntaxes}`ArmAgnosticDocs`

The `__arm_agnostic` keyword applies to prototyped function types and
affects the function's calling convention for a given state S. This
attribute allows the user to describe a function that preserves S, without
requiring the function to share S with its callers and without making
the assumption that S exists.

If a function has the `__arm_agnostic(S)` attribute and calls a function
without this attribute, then the function's object code will contain code
to preserve state S. Otherwise, the function's object code will be the same
as if it did not have the attribute.

The attribute takes string arguments to describe state S. The supported
states are:

- `"sme_za_state"` for state enabled by PSTATE.ZA, such as ZA and ZT0.

The attribute `__arm_agnostic("sme_za_state")` cannot be used in conjunction
with `__arm_in(S)`, `__arm_out(S)`, `__arm_inout(S)` or
`__arm_preserves(S)` where state S describes state enabled by PSTATE.ZA,
such as "za" or "zt0".


### __arm_in

{clang-attr-syntaxes}`ArmInDocs`

The `__arm_in` keyword applies to prototyped function types and specifies
that the function shares a given state S with its caller. For `__arm_in`, the
function takes the state S as input and returns with the state S unchanged.

The attribute takes string arguments to instruct the compiler which state
is shared. The supported states for S are:

- `"za"` for Matrix Storage (requires SME)

The attributes `__arm_in(S)`, `__arm_out(S)`, `__arm_inout(S)` and
`__arm_preserves(S)` are all mutually exclusive for the same state S.


### __arm_inout

{clang-attr-syntaxes}`ArmInOutDocs`

The `__arm_inout` keyword applies to prototyped function types and specifies
that the function shares a given state S with its caller. For `__arm_inout`,
the function takes the state S as input and returns new state for S.

The attribute takes string arguments to instruct the compiler which state
is shared. The supported states for S are:

- `"za"` for Matrix Storage (requires SME)

The attributes `__arm_in(S)`, `__arm_out(S)`, `__arm_inout(S)` and
`__arm_preserves(S)` are all mutually exclusive for the same state S.


### __arm_locally_streaming

{clang-attr-syntaxes}`ArmSmeLocallyStreamingDocs`

The `__arm_locally_streaming` keyword applies to function declarations
and specifies that all the statements in the function are executed in
streaming mode. This means that:

- the function requires that the target processor implements the Scalable Matrix
  Extension (SME).
- the program automatically puts the machine into streaming mode before
  executing the statements and automatically restores the previous mode
  afterwards.

Clang manages PSTATE.SM automatically; it is not the source code's
responsibility to do this. For example, Clang will emit code to enable
streaming mode at the start of the function, and disable streaming mode
at the end of the function.


### __arm_new

{clang-attr-syntaxes}`ArmNewDocs`

The `__arm_new` keyword applies to function declarations and specifies
that the function will create a new scope for state S.

The attribute takes string arguments to instruct the compiler for which state
to create new scope. The supported states for S are:

- `"za"` for Matrix Storage (requires SME)

For state `"za"`, this means that:

- the function requires that the target processor implements the Scalable Matrix
  Extension (SME).
- the function will commit any lazily saved ZA data.
- the function will create a new ZA context and enable PSTATE.ZA.
- the function will disable PSTATE.ZA (by setting it to 0) before returning.

For `__arm_new("za")` functions Clang will set up the ZA context automatically
on entry to the function and disable it before returning. For example, if ZA is
in a dormant state Clang will generate the code to commit a lazy-save and set up
a new ZA state before executing user code.


### __arm_out

{clang-attr-syntaxes}`ArmOutDocs`

The `__arm_out` keyword applies to prototyped function types and specifies
that the function shares a given state S with its caller. For `__arm_out`,
the function ignores the incoming state for S and returns new state for S.

The attribute takes string arguments to instruct the compiler which state
is shared. The supported states for S are:

- `"za"` for Matrix Storage (requires SME)

The attributes `__arm_in(S)`, `__arm_out(S)`, `__arm_inout(S)` and
`__arm_preserves(S)` are all mutually exclusive for the same state S.


### __arm_preserves

{clang-attr-syntaxes}`ArmPreservesDocs`

The `__arm_preserves` keyword applies to prototyped function types and
specifies that the function does not read a given state S and returns
with state S unchanged.

The attribute takes string arguments to instruct the compiler which state
is shared. The supported states for S are:

- `"za"` for Matrix Storage (requires SME)

The attributes `__arm_in(S)`, `__arm_out(S)`, `__arm_inout(S)` and
`__arm_preserves(S)` are all mutually exclusive for the same state S.


### __arm_streaming

{clang-attr-syntaxes}`ArmSmeStreamingDocs`

The `__arm_streaming` keyword applies to prototyped function types and specifies
that the function has a "streaming interface". This means that:

- the function requires that the processor implements the Scalable Matrix
  Extension (SME).
- the function must be entered in streaming mode (that is, with PSTATE.SM
  set to 1)
- the function must return in streaming mode

Clang manages PSTATE.SM automatically; it is not the source code's
responsibility to do this. For example, if a non-streaming
function calls an `__arm_streaming` function, Clang generates code
that switches into streaming mode before calling the function and
switches back to non-streaming mode on return.


### __arm_streaming_compatible

{clang-attr-syntaxes}`ArmSmeStreamingCompatibleDocs`

The `__arm_streaming_compatible` keyword applies to prototyped function types and
specifies that the function has a "streaming compatible interface". This
means that:

- the function may be entered in either non-streaming mode (PSTATE.SM=0) or
  in streaming mode (PSTATE.SM=1).
- the function must return in the same mode as it was entered.
- the code executed in the function is compatible with either mode.

Clang manages PSTATE.SM automatically; it is not the source code's
responsibility to do this. Clang will ensure that the generated code in
streaming-compatible functions is valid in either mode (PSTATE.SM=0 or
PSTATE.SM=1). For example, if an `__arm_streaming_compatible` function calls a
non-streaming function, Clang generates code to temporarily switch out of streaming
mode before calling the function and switch back to streaming-mode on return if
`PSTATE.SM` is `1` on entry of the caller. If `PSTATE.SM` is `0` on
entry to the `__arm_streaming_compatible` function, the call will be executed
without changing modes.


