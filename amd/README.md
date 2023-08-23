# AMD subprojects

This directory and its subdirectories contain source code for AMD open-source
projects which are tightly-coupled to LLVM project infrastructure, such that
there is no well-defined interface or versioning guarantees maintained which
make it useful to develop them in separate repositories.

These projects are either fundamentally unsuitable for direct contribution to
the LLVM project itself, are currently not in a state where the community is
likely to accept them directly, or AMD is not currently prepared to undergo the
upstreaming process for them. In any case, their current home in this distinct
top-level subdirectory is intended to namespace them such that there is no
possibility for conflict with upstream sources, and to leave open a path to
contributing them upstream where possible.

Most (and at the time of writing, currently all) of these projects were
originally developed in separate repositories. Their history was maintained as
parents of an octopus merge which introduced this subdirectory. A modified
script which was used to perform the merge is retained in the `utils`
subdirectory as `omnibus.sh` for historical interest, and to aid in any
external developer's transition.

Also available is a more general-purpose script,
`translate-legacy-branch-to-omnibus-monorepo.sh`, which can be used by external
developers to "translate" any branch made against the separate repositories
into a clone of llvm-project. For usage instructions please run the script with
no arguments.
