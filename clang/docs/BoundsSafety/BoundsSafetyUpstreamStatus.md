(bounds-safety-upstream-status)=
# Upstream status of `-fbounds-safety`

A preview implementation is available in the
[swiftlang/llvm-project](https://github.com/swiftlang/llvm-project/tree/stable/21.x)
fork.

`-fbounds-safety` is being upstreamed incrementally. Bounds annotations are
upstreamed first, so that projects can start annotating their code; the full
`-fbounds-safety` model follows. Until the model is complete, it is guarded by
the experimental `-cc1` flag `-fexperimental-bounds-safety`, which is not
exposed through the driver.

## Bounds annotations

| Feature | Status |
|---|---|
| `__counted_by`, `__sized_by`, `__counted_by_or_null`, `__sized_by_or_null` on struct fields | Available, with limits: the count must be a single identifier, and the annotations are handled as declaration attributes rather than type attributes |
| Use of the annotations by `-fsanitize=array-bounds` and `_FORTIFY_SOURCE=3` (`__builtin_dynamic_object_size`) | Available |
| Bounds annotations as type attributes | In progress (draft: [#224560](https://github.com/llvm/llvm-project/pull/224560)) |
| Late parsing of bounds annotations as type attributes, for struct fields | NFC groundwork merged ([#192799](https://github.com/llvm/llvm-project/pull/192799), [#223267](https://github.com/llvm/llvm-project/pull/223267), [#224556](https://github.com/llvm/llvm-project/pull/224556), [#224558](https://github.com/llvm/llvm-project/pull/224558), [#227774](https://github.com/llvm/llvm-project/pull/227774)); in review ([#224559](https://github.com/llvm/llvm-project/pull/224559)); remaining work in progress (draft: [#233583](https://github.com/llvm/llvm-project/pull/233583)) |
| `__counted_by` on function parameters, where the count refers to an earlier parameter | In progress (drafts: [#234941](https://github.com/llvm/llvm-project/pull/234941), [#234950](https://github.com/llvm/llvm-project/pull/234950), [#234963](https://github.com/llvm/llvm-project/pull/234963), [#234981](https://github.com/llvm/llvm-project/pull/234981)), contributed by Mohammed Ashraf as part of Google Summer of Code |
| Late parsing of `__counted_by` on function parameters, where the count refers to a later parameter | Planned: to be split out of [#230538](https://github.com/llvm/llvm-project/pull/230538) ([#167365](https://github.com/llvm/llvm-project/issues/167365)) |
| `__counted_by` on function return types and variables | Planned: to be split out of [#230538](https://github.com/llvm/llvm-project/pull/230538) ([#167365](https://github.com/llvm/llvm-project/issues/167365)) |
| `__single` | Planned; refactoring in progress downstream ([swiftlang/llvm-project#13651](https://github.com/swiftlang/llvm-project/pull/13651), [#13846](https://github.com/swiftlang/llvm-project/pull/13846), [#13925](https://github.com/swiftlang/llvm-project/pull/13925)), contributed by Anupam Kumar |
| `__ended_by`, `__null_terminated`, `__unsafe_indexable` | Planned (`__null_terminated`: [#230685](https://github.com/llvm/llvm-project/issues/230685)) |
| Complex count expressions | Planned ([#141827](https://github.com/llvm/llvm-project/issues/141827), [#229345](https://github.com/llvm/llvm-project/issues/229345)) |

## Full `-fbounds-safety` model

| Feature | Status |
|---|---|
| Wide pointers (`__bidi_indexable`, `__indexable`) and bounds checks | Planned |
| New AST expressions for bounds checks | Planned (RFC needed) |
| Cast and operation checks on annotated pointers | Planned |
| Implicit (default) bounds annotations | Planned |
| Checks on assignments to a pointer and its count in the same scope | Planned (RFC needed) |

## Roadmap

- Early 2027: bounds annotations are ready for open-source projects to adopt.
- After that: the full `-fbounds-safety` model.

## Tracking

- [Open issues](https://github.com/llvm/llvm-project/issues?q=is%3Aopen+label%3Aclang%3Abounds-safety)
- [Open pull requests](https://github.com/llvm/llvm-project/pulls?q=is%3Aopen+label%3Aclang%3Abounds-safety)
