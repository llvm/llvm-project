# Adding Target Support

LLDB supports many combinations of architecture, operating system and other
system components. In this document we describe the considerations and
requirements for porting LLDB to a combination of those things which we will
refer to as a "target" within this document.

:::{note}
The terms `Target` and `Platform` are used throughout LLDB, often
referring to a subset of what this document calls a "target".
Unfortunately there is no more specific word to use here.
:::

This document provides some hints on implementation, but because every target
is unique, we expect developers to learn about implementation from existing
targets.

This document starts with the end of the process, proposing your target for
upstream inclusion. It is worth reading through this even if that is not one of
your goals, as you will have to tackle the same topics even in a downstream
implementation.

## Upstreaming Proposal Process

You must send an RFC to the
[LLDB Discourse forum](https://discourse.llvm.org/c/subprojects/lldb/8) before
upstreaming new target support. This RFC must be accepted in some form by the
community before any changes can be merged upstream that are specific to your
target.

This RFC follows the normal rules of the LLVM community decision making process.

We require an RFC for target support upstreaming because:
* We want to ensure there is some level of public discussion on the topic.
  It is important that details, even those obvious to the community at the time,
  are written down so that future readers may learn from them.
* These discussions form the basis for future proposals and the assessment
  of those proposals. Every proposal will be different in some way, and by
  contrasting current proposals with the previous proposals we can do a better
  job assessing them.

RFCs are not required to be:
* In a set format. Make a logical argument in whatever way you think fit.
* Exhaustively detailed. Include what you think is relevant and the community
  will ask for the rest.
* Answering all the same questions, or making all the same points, as previous
  proposals. Compare your target with existing targets, but not everything will,
  or has to, apply to yours.
* A commitment to the maximum theoretical level of support (more on this later).
  If your resources are limited, say so, and that will be taken into account
  (and vice versa, committing to a lot of work does not guarantee acceptance).

## Removal Proposal Process

Removing target support also requires an RFC. Reasonable effort should be made
to involve the original contributors and users of the target support.

:::{note}
You may propose the removal of a subset of the target. For instance, removing
support for an operating system on architecture A, without removing support for
that operating system on architecture B.
:::

The content of the RFC should cover the same topics as an upstreaming proposal,
but you will be arguing for the opposite conclusion.

We reserve the right to change our standards over time, so a removal proposal is
not necessarily a refutation of each point made in the upstreaming proposal.
For example, what was an acceptable level of testing in the past may no longer
be acceptable.

If standards change and are to be applied to existing targets, target
stakeholders should be given the opportunity to meet those standards rather
than going straight to removal.

## Expectations Of Upstream Code

Listed below are some examples of factors considered when considering accepting
code upstream. These are examples and RFC authors are free to add their own,
leave some out, or explain why they do not apply to their proposal.

If you do use these points, they need to come with an answer and evidence to
justify the answer, rather than simply "yes this applies to my target".
In other words, your proposal must stand alone without requiring readers to read
this document as well.

The first set covers your motivation for your target being upstream, rather
than on a fork:

* Will it help you distribute an LLDB that includes this target support?
  For example if there is an existing community for this
  target, and how would they acquire LLDB?
* Will it enable a wider community than your own forks would?
  For example for it to be included in Linux distribution packaging.
* Will it improve support for other targets by being there?
  For example if we already support Operating System X on Architecture Y,
  adding Architecture Z support may improve both as a side effect.
* Will it help you keep your changes in sync?
  For example if it involves fundamental changes to LLDB, or you have a very
  small amount of maintenance resources in your community.
* What other costs (or benefits) do you incur staying on a fork?
  For example, your company might already have a fork.

The next set is about whether you, your community, or the LLDB community, can
adequately maintain the code upstream:

* Who will be the maintainers for this target? Ideally there will be more than
  one, who is present in the LLVM community and can be contacted
  in a few different ways.
* How often will it be tested, where, by whom and who will be responsible for
  it?
* Who will address problems with it? Will it always be the named maintainers?
  Is it so common that anyone in upstream LLDB can deal with it, or will 
  only employees of a specific company work on it.
* When it breaks, how easy will it be for the upstream LLDB community to
  continue their work without disruption?
* If upstream contributors want to reproduce issues on your target, how can
  they access it? Can it be emulated or virtualized? Does it require them to
  sign a license? Do you offer access for open source projects? (and does that
  include employees of other companies)

## The Extent of Target Support

The factors above are part of a cost benefit analysis that the LLDB community
will do, prioritising the health of the community and the project. This means
that there is no set level or type of commitment required for a new target.

Each case will have unique aspects, so you need to tell the community how much
impact your target will have on LLDB. Below are some questions you can start
with, along with stereotypical "big" and "small" answers.

Note that these are intentionally not maximums and minimums. The impact
of some targets will be bigger than "big" or smaller than "small".

* How many users will use LLDB with this target?
  * Big: millions of developers worldwide.
  * Small: you and a small community of target users.
* How many changes will it require?
  * Big: large changes to all parts of LLDB. On the level of the existing
    code for C and C++ support.
  * Small: small changes to enable existing support from LLVM, changes
    to packet parsing (see the MSP430 case study below).
* What parts and features of LLDB will work on, or with, this target?
  * Big: `lldb`, `lldb-server` and advanced features like shared libraries.
  * Small: just `lldb` and only basic features like continue, stop, reading
    memory and registers.
* How often is it tested?
  * Big: per-commit testing of LLDB, following the upstream llvm-project.
  * Small: per release of a downstream community or individual developer's
    tools.
* Who will maintain it?
  * Big: there are several listed maintainers for this target, who are
    employed by a company with significant investments in the target.
  * Small: there is a single maintainer listed for this target.
* When it breaks, who will be affected?
  * Big: every single developer and user of LLDB.
  * Small: you and your target's community.
* How do I reproduce a problem on this target?
  * Big: all components are open source and can be run anywhere by using
    simulations with very little setup.
  * Small: you contact the maintainer and they do it for you.
* Who will fix problems?
  * Big: the maintainers and a large group of contributors.
  * Small: only the listed maintainer.

## Target Support Case Studies

These case studies are to give you starting points for your proposals. Consider
how your target compares to these.

### Apple Targets

Apple targets have possibly the most extensive support in LLDB.

* Changes from LLVM's main branch are continuously tested. Results are
  accessible publicly, and reported to all LLDB contributors.
* Many employees contribute upstream and are maintainers for these targets.
* Target-specific problems are either solved by the upstream community with
  maintainer input, or by the maintainers themselves.
* Apple-specific features either do not impact other targets, or, when they
  do, they are designed and maintained in collaboration with the community.
* Many people unrelated to Apple itself use Apple hardware, which is
  available at retail. So it is fairly easy to find someone who can reproduce
  an issue.
* LLDB gets a lot of secondary testing downstream, and exposure to Apple's
  own developer community.

### FreeBSD

FreeBSD is an example of quite self-contained support, managed by the project's
community.

* FreeBSD is open source and available to anyone to build, modify, run on
  hardware, emulate or virtualize.
* FreeBSD support is mostly in the native target parts of `lldb-server`,
  so issues with it rarely impact any other target.
* It is quite similar to Linux, so it does not cause large scale changes to
  fundamental assumptions LLDB makes.
* It has at least one maintainer and sometimes contributions to upstream LLDB
  from the FreeBSD community.
* Issues are solved by the maintainer and the FreeBSD community.
* LLDB on FreeBSD is tested each time FreeBSD updates the version used in the
  base system, which is roughly per upstream LLVM release.

### Linux

Linux is an example where the Linux community as a whole does not do all the
work of Linux support. Some architectures have a wide contributor base and
others have company-specific contributors as well.

* Linux is open source, freely available, and can be installed on hardware,
  virtualised, emulated, and so on. Most problems can be reproduced in more than
  one way.
* Many architectures are tested by the architecture vendor, others
  are tested on community-provided hardware.
* LLDB is a commonly available Linux package, so further testing happens during
  various Linux distributions' release cycles.
* Support for a new architecture is quite easy to isolate, so problems do not
  impact other Linux architectures, or other operating systems.
* Linux is very popular, therefore its choice of standards informs many core
  features of LLDB.
* It has more than one listed maintainer and many contributors.
  Issues are resolved by the community.

### MSP430

MSP430 is handled as a bare-metal (no operating system) target in LLDB. So it
is the most minimal example.

* LLVM already has MSP430 support, so changes to enable it in LLDB were minimal.
* It does not use `lldb-server`, and `lldb` only required minor adjustments to
  be compatible with the commonly used debug server.
* An ABI plugin was added, which is isolated to MSP430 only.
* It is not systematically tested anywhere, but gets some use by MSP430
  developers.
* It has no documented maintainer.
* We rely on users to report issues with it, and would likely guide them to fix
  them themselves.

## Components Of Target Support

This is a very high-level view. We recommend you combine this with reading the
changes made for recently added targets, as each target is going to be slightly
different.

Assuming that:
* You want to support a combination of a new architecture and a new operating
  system.
* LLVM already supports your architecture and operating system.
* You want to port most of the LLDB features, which means porting both `lldb`
  and `lldb-server`.

Then this is a list of components you will need to write. If your target is
similar to others you may be able to reuse existing components and we encourage
you to do so.

First the components concerned with operating systems:

* A `Platform` plugin. For example `PlatformLinux`.
* A `HostInfo` plugin. For example `HostInfoLinux`.
* A native process plugin (native means it runs on your target). For example
  `NativeProcessLinux`.
* Signal information. For example `LinuxSignals`.
* A dynamic loader plugin. For example `DynamicLoaderPOSIXDYLD`.
* An object file plugin. For example `ObjectFileELF`.

Then the components concerned with architecture (though the distinction is often
inexact):

* ABI plugin. For example `ABISysV_arm64`.
* Architecture plugin. For example `ArchitectureAArch64`.
* Register definitions and register context (a context is a collection of
  registers). For example `NativeRegisterContextLinux_arm64`.
* Unwinding support for backtracing. LLDB needs to be taught about any
  architecture-specific directives.
* Instruction emulation. For example `EmulateInstructionARM64`. Most targets
  need to emulate a small number of instructions for unwinding. If your target
  does not have hardware single-step, you will need to emulate many times more
  than that.

The order of implementation will vary in each case. Each component does not
need to be fully implemented for you to start work on the next.

If you are going to be running `lldb` on your target, then an obvious first step
is to get `lldb` to build there without any changes. This `lldb` will only be
useful for remotely debugging other targets, but you will at least know that
the build system is compatible.

Next, try `lldb` with any existing debug servers for your target. If they
are similar to `lldb-server` or `gdbserver`, this can flush out some
obvious issues in `lldb`.

Then begin porting `lldb-server` to your target. We recommend that you
get the test suite to run as soon as possible, however bad the results
are.

As you have seen above, there are a lot of moving parts to a debugger. So
having some set of results to measure progress is very important.
Sometimes a change will get one test to pass, sometimes hundreds, and it
is easy to regress if you are not careful.
