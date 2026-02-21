# Clang Maintainers

This file is a list of the
[maintainers](https://llvm.org/docs/DeveloperPolicy.html#maintainers)
for Clang. The list of current Clang Area Team members can be found
[here](https://github.com/llvm/llvm-project/blob/main/clang/AreaTeamMembers.txt).

```{contents} Table of Contents
:depth: 2
```

# Active Maintainers

The following people are the active maintainers for the project. Please
reach out to them for code reviews, questions about their area of
expertise, or other assistance.

## Lead Maintainer

Aaron Ballman \
aaron@aaronballman.com (email), aaron.ballman (Phabricator), [AaronBallman](https://github.com/AaronBallman) (GitHub), AaronBallman (Discourse), aaronballman (Discord)

## Contained Components

These maintainers are responsible for particular high-level components
within Clang that are typically contained to one area of the compiler.

### AST matchers

Aaron Ballman \
aaron@aaronballman.com (email), aaron.ballman (Phabricator), [AaronBallman](https://github.com/AaronBallman) (GitHub), AaronBallman (Discourse), aaronballman (Discord)

### AST Visitors

Sirraide \
aeternalmail@gmail.com (email), [Sirraide](https://github.com/Sirraide) (GitHub), Ætérnal (Discord), Sirraide (Discourse)

### Clang LLVM IR generation

Eli Friedman \
efriedma@qti.qualcomm.com (email), efriedma (Phabricator), [efriedma-quic](https://github.com/efriedma-quic) (GitHub)

Anton Korobeynikov \
anton@korobeynikov.info (email), asl (Phabricator), [asl](https://github.com/asl) (GitHub)

### Clang MLIR generation

Andy Kaylor \
akaylor@nvidia.com (email), AndyKaylor (Discord), [AndyKaylor](https://github.com/AndyKaylor) (GitHub)

Bruno Cardoso Lopes \
bruno.cardoso@gmail.com (email), sonicsprawl (Discord), [bcardosolopes](https://github.com/bcardosolopes) (GitHub)

Henrich Lauko \
henrich.lau@gmail.com (email), henrich.lauko (Discord), [xlauko](https://github.com/xlauko) (GitHub)

### Analysis & CFG

Yitzhak Mandelbaum \
yitzhakm@google.com (email), ymandel (Phabricator), [ymand](https://github.com/ymand) (GitHub)

### Sema

Sirraide \
aeternalmail@gmail.com (email), [Sirraide](https://github.com/Sirraide) (GitHub), Ætérnal (Discord), Sirraide (Discourse)

Mariya Podchishchaeva \
mariya.podchishchaeva@intel.com (email), [Fznamznon](https://github.com/Fznamznon) (GitHub), fznamznon (Discord), Fznamznon (Discourse)

### Recovery AST

Haojian Wu \
hokein.wu@gmail.com (email), hokein (Phabricator), [hokein](https://github.com/hokein) (GitHub), hokein (Discourse)

### Experimental new constant interpreter

Timm Bäder \
tbaeder@redhat.com (email), tbaeder (Phabricator), [tbaederr](https://github.com/tbaederr) (GitHub), tbaeder (Discourse), tbaeder (Discord)

### Modules & serialization

Chuanqi Xu \
yedeng.yd@linux.alibaba.com (email), ChuanqiXu (Phabricator), [ChuanqiXu9](https://github.com/ChuanqiXu9) (GitHub)

Michael Spencer \
bigcheesegs@gmail.com (email), Bigcheese (Phabricator), [Bigcheese](https://github.com/Bigcheese) (GitHub)

Vassil Vassilev \
Vassil.Vassilev@cern.ch (email), v.g.vassilev (Phabricator), [vgvassilev](https://github.com/vgvassilev) (GitHub)

### Templates

Erich Keane \
ekeane@nvidia.com (email), ErichKeane (Phabricator), [erichkeane](https://github.com/erichkeane) (GitHub)

### Concepts

Corentin Jabot \
corentin.jabot@gmail.com (email), cor3ntin (Phabricator), [cor3ntin](https://github.com/cor3ntin) (GitHub)

### Lambdas

Corentin Jabot \
corentin.jabot@gmail.com (email), cor3ntin (Phabricator), [cor3ntin](https://github.com/cor3ntin) (GitHub)

### Debug information

Adrian Prantl \
aprantl@apple.com (email), aprantl (Phabricator), [adrian-prantl](https://github.com/adrian-prantl) (GitHub)

David Blaikie \
dblaikie@gmail.com (email), dblaikie (Phabricator), [dwblaikie](https://github.com/dwblaikie) (GitHub)

Eric Christopher \
echristo@gmail.com (email), echristo (Phabricator), [echristo](https://github.com/echristo) (GitHub)

### Exception handling

Anton Korobeynikov \
anton@korobeynikov.info (email), asl (Phabricator), [asl](https://github.com/asl) (GitHub)

### Clang static analyzer

Artem Dergachev \
artem.dergachev@gmail.com (email), NoQ (Phabricator), [haoNoQ](https://github.com/haoNoQ) (GitHub)

Gábor Horváth \
xazax.hun@gmail.com (email), xazax.hun (Phabricator), [Xazax-hun](https://github.com/Xazax-hun) (GitHub)

Balázs Benics \
benicsbalazs@gmail.com (email), steakhal (Phabricator), [steakhal](https://github.com/steakhal) (GitHub)

Donát Nagy \
donat.nagy@ericsson.com (email), [NagyDonat](https://github.com/NagyDonat) (GitHub), DonatNagyE (Discourse)

### Compiler options

Jan Svoboda \
jan_svoboda@apple.com (email), jansvoboda11 (Phabricator), [jansvoboda11](https://github.com/jansvoboda11) (GitHub)

### API Notes

Egor Zhdan \
e_zhdan@apple.com (email), [egorzhdan](https://github.com/egorzhdan) (GitHub), egor.zhdan (Discourse)

Saleem Abdulrasool \
compnerd@compnerd.org (email), [compnerd](https://github.com/compnerd) (GitHub), compnerd (Discourse)

### OpenBSD driver

Brad Smith \
brad@comstyle.com (email), brad (Phabricator), [brad0](https://github.com/brad0) (GitHub)

### Offloading driver

Joseph Huber \
joseph.huber@amd.com (email), [jhuber6](https://github.com/jhuber6) (GitHub)

Nick Sarnie \
nick.sarnie@intel.com (email), [sarnex](https://github.com/sarnex) (GitHub)

### Driver parts not covered by someone else

Fangrui Song \
i@maskray.me (email), MaskRay (Phabricator), [MaskRay](https://github.com/MaskRay) (GitHub)

### Constant Expressions

Mariya Podchishchaeva \
mariya.podchishchaeva@intel.com (email), [Fznamznon](https://github.com/Fznamznon) (GitHub), fznamznon (Discord), Fznamznon (Discourse)

### Thread Safety Analysis

Aaron Puchert \
aaron.puchert@sap.com (email), [aaronpuchert](https://github.com/aaronpuchert) (GitHub), aaronpuchert (Discourse)

### Function Effect Analysis

Doug Wyatt \
dwyatt@apple.com (email), [dougsonos](https://github.com/dougsonos) (GitHub), dougsonos (Discourse)

Sirraide \
aeternalmail@gmail.com (email), [Sirraide](https://github.com/Sirraide) (GitHub), Ætérnal (Discord), Sirraide (Discourse)

### Code Coverage

Takumi Nakamura \
geek4civic@gmail.com (email), chapuni(GitHub), chapuni (Discord), chapuni (Discourse)

Alan Phipps \
a-phipps@ti.com (email), [evodius96](https://github.com/evodius96) (GitHub), evodius96 (Discourse)

### Python Bindings

Vlad Serebrennikov \
serebrennikov.vladislav@gmail.com (email), [Endilll](https://github.com/Endilll) (GitHub), Endill (Discord), Endill (Discourse)

## Tools

These maintainers are responsible for user-facing tools under the Clang
umbrella or components used to support such tools.

### clang-format

MyDeveloperDay \
mydeveloperday@gmail.com (email), MyDeveloperDay (Phabricator), [MyDeveloperDay](https://github.com/MyDeveloperDay) (GitHub)

Owen Pan \
owenpiano@gmail.com (email), owenpan (Phabricator), [owenca](https://github.com/owenca) (GitHub)

## ABIs

The following people are responsible for decisions involving ABI.

### Itanium ABI

### Microsoft ABI

Reid Kleckner \
rnk@llvm.org (email), [rnk](https://github.com/rnk) (GitHub), rnk (Discourse), rnk (Discord), rnk (Phabricator)

### ARM EABI

Anton Korobeynikov \
anton@korobeynikov.info (email), asl (Phabricator), [asl](https://github.com/asl) (GitHub)

## Compiler-Wide Topics

The following people are responsible for functionality that does not fit
into a single part of the compiler, but instead spans multiple
components within the compiler.

### Attributes

Aaron Ballman \
aaron@aaronballman.com (email), aaron.ballman (Phabricator), [AaronBallman](https://github.com/AaronBallman) (GitHub), AaronBallman (Discourse), aaronballman (Discord)

### Plugins

Vassil Vassilev \
Vassil.Vassilev@cern.ch (email), v.g.vassilev (Phabricator), [vgvassilev](https://github.com/vgvassilev) (GitHub)

### Inline assembly

Eric Christopher \
echristo@gmail.com (email), echristo (Phabricator), [echristo](https://github.com/echristo) (GitHub)

### Text encodings

Corentin Jabot \
corentin.jabot@gmail.com (email), cor3ntin (Phabricator), [cor3ntin](https://github.com/cor3ntin) (GitHub)

### CMake integration

Petr Hosek \
phosek@google.com (email), phosek (Phabricator), [petrhosek](https://github.com/petrhosek) (GitHub)

### General Windows support

Reid Kleckner \
rnk@llvm.org (email), [rnk](https://github.com/rnk) (GitHub), rnk (Discourse), rnk (Discord), rnk (Phabricator)

### Incremental compilation, REPLs, clang-repl

Vassil Vassilev \
Vassil.Vassilev@cern.ch (email), v.g.vassilev (Phabricator), [vgvassilev](https://github.com/vgvassilev) (GitHub)

## Standards Conformance

The following people are responsible for validating that changes are
conforming to a relevant standard. Contact them for questions about how
to interpret a standard, when fixing standards bugs, or when
implementing a new standard feature.

### C conformance

Aaron Ballman \
aaron@aaronballman.com (email), aaron.ballman (Phabricator), [AaronBallman](https://github.com/AaronBallman) (GitHub), AaronBallman (Discourse), aaronballman (Discord)

### C++ conformance

Hubert Tong \
hubert.reinterpretcast@gmail.com (email), hubert.reinterpretcast (Phabricator), [hubert-reinterpretcast](https://github.com/hubert-reinterpretcast) (GitHub)

Shafik Yaghmour \
shafik.yaghmour@intel.com (email), [shafik](https://github.com/shafik) (GitHub), shafik.yaghmour (Discord), shafik (Discourse)

Vlad Serebrennikov \
serebrennikov.vladislav@gmail.com (email), [Endilll](https://github.com/Endilll) (GitHub), Endill (Discord), Endill (Discourse)

### C++ Defect Reports

Vlad Serebrennikov \
serebrennikov.vladislav@gmail.com (email), [Endilll](https://github.com/Endilll) (GitHub), Endill (Discord), Endill (Discourse)

### Objective-C/C++ conformance

Akira Hatanaka \
ahatanak@gmail.com, [ahatanak](https://github.com/ahatanak) (GitHub), ahatanak4220 (Discord), ahatanak (Discourse)

### OpenMP conformance

Alexey Bataev \
a.bataev@hotmail.com (email), ABataev (Phabricator), [alexey-bataev](https://github.com/alexey-bataev) (GitHub)

### OpenCL conformance

Sven van Haastregt \
sven.vanhaastregt@arm.com (email), [svenvh](https://github.com/svenvh) (GitHub)

### OpenACC

Erich Keane \
ekeane@nvidia.com (email), ErichKeane (Phabricator), [erichkeane](https://github.com/erichkeane) (GitHub)

### SYCL conformance

Alexey Bader \
alexey.bader@intel.com (email), bader (Phabricator), [bader](https://github.com/bader) (GitHub)

### HLSL conformance

Chris Bieneman \
chris.bieneman@gmail.com (email), [llvm-beanz](https://github.com/llvm-beanz) (GitHub), beanz (Discord), beanz (Discourse)

### Issue Triage

Shafik Yaghmour \
shafik.yaghmour@intel.com (email), [shafik](https://github.com/shafik) (GitHub), shafik.yaghmour (Discord), shafik (Discourse)

hstk30 \
hanwei62@huawei.com (email), [hstk30-hw](https://github.com/hstk30-hw) (GitHub), hstk30(Discord), hstk30 (Discourse)

# Inactive Maintainers

The following people have graciously spent time performing
maintainership responsibilities but are no longer active in that role.
Thank you for all your help with the success of the project!

## Emeritus Lead Maintainers

Doug Gregor (dgregor@apple.com) \
Richard Smith (richard@metafoo.co.uk)

## Inactive component maintainers

Anastasia Stulova (stulovaa@gmail.com) \-- OpenCL, C++ for OpenCL \
Chandler Carruth (chandlerc@gmail.com, chandlerc@google.com) \--   CMake, library layering \
Devin Coughlin (dcoughlin@apple.com) \-- Clang static analyzer \
Manuel Klimek (klimek@google.com (email), klimek (Phabricator), [r4nt](https://github.com/r4nt) (GitHub)) \-- Tooling, AST matchers \
Dmitri Gribenko (gribozavr@gmail.com (email), gribozavr (Phabricator), [gribozavr](https://github.com/gribozavr) (GitHub)) \-- Analysis & CFG \
Tom Honermann (tom@honermann.net (email), tahonermann (Phabricator), [tahonermann](https://github.com/tahonermann) (GitHub)) \-- Text Encodings \
John McCall (rjmccall@apple.com (email), rjmccall (Phabricator), [rjmccall](https://github.com/rjmccall) (GitHub)) \-- Clang LLVM IR generation, Objective-C/C++ conformance, Itanium ABI \
John Ericson (git@johnericson.me (email), Ericson2314 (Phabricator), [Ericson2314](https://github.com/Ericson2314) (GitHub)) \-- CMake Integration \
Stanislav Gatev (sgatev@google.com (email), sgatev (Phabricator), [sgatev](https://github.com/sgatev) (GitHub)) -- Analysis & CFG
