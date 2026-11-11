(design_philosophy)=

# Design Philosophy

This page describes what LLVM-libc is for, who it serves, and what it
deliberately does not try to be. Most of the rules elsewhere in the developer
documentation follow from the decisions recorded here. Read this page before
starting a change whose goal is "make program X work with LLVM-libc", because
the easiest way to reach that goal is often the wrong one for the project.

## An application libc

Most C libraries are *system* libcs. One shared copy is installed on a machine,
every program on that machine uses it, and so its ABI and observable behavior
are very hard to change. LLVM-libc is designed as an *application* libc: it is
built for a specific program and linked into it.

This has consequences throughout the project:

- **Static linking is the primary use case.** Dynamic loading is under
  development because features such as `dlopen` need it, but it is not the
  primary use case and should not drive design decisions.
- **There is no stable ABI.** The library and the program are built together,
  so type layouts and internal symbols are free to change. See
  {ref}`abi_policy`.
- **The library is configured for its user.** Each target selects its own set
  of entrypoints, configuration options, and generated headers, instead of the
  library shipping one build that tries to suit everyone.
- **Users track `main`.** There is no separate release stream for LLVM-libc
  today. This is likely to change when the library is shipped more widely.

Writing a new libc is an opportunity to design an API and ABI for modern
systems. Preserving the freedom to do that is worth more to the project than
making any one program work sooner.

## Who we build for

LLVM-libc has several kinds of users: statically linked programs on Linux
servers, baremetal embedded systems, GPUs, other LLVM projects that use its
internals through {ref}`Hand-in-Hand <hand_in_hand>`, and general open-source
software. Their needs differ. In general, baremetal users prioritize binary
size, for example, while server users prioritize latency and throughput.

The first choice is always a design that serves every user. When that is not
possible, a feature added for one group of users must not impose a cost on
the others that they cannot configure away.

The longer-term milestones for general software support are building web
browsers (Chromium and Firefox) and the generic Linux utilities needed for a
complete Linux system built with LLVM and no GNU components.

## Non-goals

- **Compatibility with glibc, or any other libc, as a goal in itself.**
  LLVM-libc implements the C and POSIX standards. It does not pretend to be
  another implementation. See {ref}`making_existing_software_work` for what
  this means in practice.
- **Bug-for-bug compatibility.** Where a standard defines the behavior,
  LLVM-libc follows the standard even if a widely used libc does not. See
  {ref}`undefined_behavior`.
- **32-bit `time_t`.** `time_t` is 64 bits wide on every target.
- **"POSIX everywhere".** On platforms that are not POSIX systems, the API
  should be shaped for the target and not be an emulation of Linux.

The following are *not* non-goals. They are missing or incomplete today:

| Feature | Status |
| --- | --- |
| Dynamic loading | Under development. Never intended to be the primary use case. |
| Time zone database | Planned. The intent is to base the implementation on the one in libc++. |
| Thread cancellation | Low priority. When added, it must not significantly slow down code that does not use it. |
| Message catalogs, Annex K | Low priority. |
| Locales | See the next section. |

## Text, encodings, and locales

The locale design in the C standard conflates two different kinds of text:
text exchanged with other programs (file formats, protocols, command lines) and
text shown to or read from a person. Modern software needs to treat these
differently, and a process-wide locale that silently changes the behavior of
`printf` or `strtod` makes the first kind unreliable.

LLVM-libc's direction is:

- **Functions without an explicit locale argument ignore locales.** They
  assume their input or output is exchanged with another program.
- **`char` holds the platform's native encoding.** That is usually ASCII
  compatible, but it may be EBCDIC: libc++ supports IBM targets, and LLVM-libc
  supports libc++. Code must not assume ASCII values or that letters are
  contiguous.
- **Multibyte strings are UTF-8.**
- **`wchar_t` strings are UTF-32 or UTF-16**, depending on the target. UTF-16
  support is in progress.
- **Encoding assumptions are centralized.** They live in
  `src/__support/ctype_utils.h`, `src/__support/wctype_utils.h`, and the
  converters in `src/__support/wchar/`. All code uses those helpers instead of
  doing arithmetic on characters, including greater/less than comparisons.
- **Text for people goes through explicit-locale functions.** The intent is to
  provide the `_l` variants of the relevant functions, so that a caller who is
  formatting text for display passes a locale explicitly.

Today only the "C" locale is implemented. A function that depends on a locale
implements the "C" locale behavior and does not claim to do more.

(making_existing_software_work)=

## Making existing software work

Building real programs against LLVM-libc is how missing pieces are found, and
those reports are welcome. What matters is how each gap is closed. In order of
preference:

1. **Fix LLVM-libc where it deviates from a standard.** A wrong return value
   or a missing error condition is simply a bug.
2. **Implement the missing standard function, macro, type, or header.**
3. **Add transitive includes in public headers that POSIX requires** when a
   real program relies on them.
4. **Add a widely used extension** (GNU, BSD, or Linux specific) when a real
   program needs it. See {ref}`extensions_and_stubs`.
5. **Accommodate a non-standard assumption** only when doing so is cheap and
   does not limit future changes to LLVM-libc. Declaring `alloca` from
   `<stdlib.h>` is an example of an acceptable accommodation.
6. **Change the program, or how it is configured,** when it depends on the
   internals of another libc or on behavior no standard promises.

The following are never acceptable ways to make a program work:

- Changing the layout or kind of a public type solely to match another libc. For
  example, `pthread_t` is a structure in LLVM-libc. POSIX allows that, and code
  that treats `pthread_t` as an integer or a pointer is not portable. Making it
  a scalar would remove the freedom to change it later.
- Changing observable behavior to match another libc's deviation from a
  standard.
- Adding another libc's private symbols so that objects compiled against its
  headers link. The one narrow exception is described in {ref}`abi_policy`.

If a task can only be completed by doing one of these, stop and raise it with
the maintainers instead.

(extensions_and_stubs)=

## Extensions, stubs, and experimental entrypoints

Before implementing a function that is not in a standard, ask whether anyone
needs it. An extension needs a specific user: a named program or library that
calls it.

A function that is declared and linkable but not fully implemented goes behind
the `LLVM_LIBC_ENABLE_EXPERIMENTAL_ENTRYPOINTS` CMake option. The bar for
adding a stub there is low, with two conditions:

- It has a specific user, as above.
- A call that the stub cannot handle fails clearly. A stub that silently
  reports success hides the missing functionality from the people who would
  otherwise report or fix it.

An entrypoint leaves the experimental list when it implements all of the
functionality its specification requires. Some incomplete entrypoints predate
the option and are still in the default lists; they should move.

## Original implementations only

Do not copy code from other libc implementations, and do not port it line by
line. This is a licensing requirement, and it is also a question of style:
LLVM-libc is written in idiomatic C++, while most other libcs are written in
C. Code translated from C rarely fits the conventions described in the
{ref}`developer guides <developer_guides>`.

Standards documents, manual pages, and observing the behavior of another
implementation as a black box are all appropriate references.

If you own an existing implementation of a libc function that you would like to
contribute, that is likely fine assuming it is updated to match the LLVM-libc
style and placed under the LLVM license. Reach out to the maintainers for more
information.
