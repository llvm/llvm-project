.. title:: clang-tidy - Clang-Tidy Checks

Clang-Tidy Checks
=================

.. toctree::
   :glob:
   :hidden:

   abseil/*
   altera/*
   android/*
   boost/*
   bugprone/*
   cert/*
   clang-analyzer/*
   concurrency/*
   cppcoreguidelines/*
   darwin/*
   fuchsia/*
   google/*
   hicpp/*
   linuxkernel/*
   llvm/*
   llvmlibc/*
   misc/*
   modernize/*
   mpi/*
   objc/*
   openmp/*
   performance/*
   portability/*
   readability/*
   zircon/*

.. csv-table::
   :header: "Name", "Offers fixes"

   :doc:`abseil-cleanup-ctad <abseil/cleanup-ctad>`, "Yes"
   :doc:`abseil-duration-addition <abseil/duration-addition>`, "Yes"
   :doc:`abseil-duration-comparison <abseil/duration-comparison>`, "Yes"
   :doc:`abseil-duration-conversion-cast <abseil/duration-conversion-cast>`, "Yes"
   :doc:`abseil-duration-division <abseil/duration-division>`, "Yes"
   :doc:`abseil-duration-factory-float <abseil/duration-factory-float>`, "Yes"
   :doc:`abseil-duration-factory-scale <abseil/duration-factory-scale>`, "Yes"
   :doc:`abseil-duration-subtraction <abseil/duration-subtraction>`, "Yes"
   :doc:`abseil-duration-unnecessary-conversion <abseil/duration-unnecessary-conversion>`, "Yes"
   :doc:`abseil-faster-strsplit-delimiter <abseil/faster-strsplit-delimiter>`, "Yes"
   :doc:`abseil-no-internal-dependencies <abseil/no-internal-dependencies>`,
   :doc:`abseil-no-namespace <abseil/no-namespace>`,
   :doc:`abseil-redundant-strcat-calls <abseil/redundant-strcat-calls>`, "Yes"
   :doc:`abseil-str-cat-append <abseil/str-cat-append>`, "Yes"
   :doc:`abseil-string-find-startswith <abseil/string-find-startswith>`, "Yes"
   :doc:`abseil-string-find-str-contains <abseil/string-find-str-contains>`, "Yes"
   :doc:`abseil-time-comparison <abseil/time-comparison>`, "Yes"
   :doc:`abseil-time-subtraction <abseil/time-subtraction>`, "Yes"
   :doc:`abseil-upgrade-duration-conversions <abseil/upgrade-duration-conversions>`, "Yes"
   :doc:`altera-id-dependent-backward-branch <altera/id-dependent-backward-branch>`,
   :doc:`altera-kernel-name-restriction <altera/kernel-name-restriction>`,
   :doc:`altera-single-work-item-barrier <altera/single-work-item-barrier>`,
   :doc:`altera-struct-pack-align <altera/struct-pack-align>`, "Yes"
   :doc:`altera-unroll-loops <altera/unroll-loops>`,
   :doc:`android-cloexec-accept <android/cloexec-accept>`, "Yes"
   :doc:`android-cloexec-accept4 <android/cloexec-accept4>`, "Yes"
   :doc:`android-cloexec-creat <android/cloexec-creat>`, "Yes"
   :doc:`android-cloexec-dup <android/cloexec-dup>`, "Yes"
   :doc:`android-cloexec-epoll-create <android/cloexec-epoll-create>`, "Yes"
   :doc:`android-cloexec-epoll-create1 <android/cloexec-epoll-create1>`, "Yes"
   :doc:`android-cloexec-fopen <android/cloexec-fopen>`, "Yes"
   :doc:`android-cloexec-inotify-init <android/cloexec-inotify-init>`, "Yes"
   :doc:`android-cloexec-inotify-init1 <android/cloexec-inotify-init1>`, "Yes"
   :doc:`android-cloexec-memfd-create <android/cloexec-memfd-create>`, "Yes"
   :doc:`android-cloexec-open <android/cloexec-open>`, "Yes"
   :doc:`android-cloexec-pipe <android/cloexec-pipe>`, "Yes"
   :doc:`android-cloexec-pipe2 <android/cloexec-pipe2>`, "Yes"
   :doc:`android-cloexec-socket <android/cloexec-socket>`, "Yes"
   :doc:`android-comparison-in-temp-failure-retry <android/comparison-in-temp-failure-retry>`,
   :doc:`boost-use-to-string <boost/use-to-string>`, "Yes"
   :doc:`bugprone-argument-comment <bugprone/argument-comment>`, "Yes"
   :doc:`bugprone-assert-side-effect <bugprone/assert-side-effect>`,
   :doc:`bugprone-assignment-in-if-condition <bugprone/assignment-in-if-condition>`,
   :doc:`bugprone-bad-signal-to-kill-thread <bugprone/bad-signal-to-kill-thread>`,
   :doc:`bugprone-bool-pointer-implicit-conversion <bugprone/bool-pointer-implicit-conversion>`, "Yes"
   :doc:`bugprone-branch-clone <bugprone/branch-clone>`,
   :doc:`bugprone-casting-through-void <bugprone/casting-through-void>`,
   :doc:`bugprone-chained-comparison <bugprone/chained-comparison>`,
   :doc:`bugprone-compare-pointer-to-member-virtual-function <bugprone/compare-pointer-to-member-virtual-function>`,
   :doc:`bugprone-copy-constructor-init <bugprone/copy-constructor-init>`, "Yes"
   :doc:`bugprone-crtp-constructor-accessibility <bugprone/crtp-constructor-accessibility>`, "Yes"
   :doc:`bugprone-dangling-handle <bugprone/dangling-handle>`,
   :doc:`bugprone-dynamic-static-initializers <bugprone/dynamic-static-initializers>`,
   :doc:`bugprone-easily-swappable-parameters <bugprone/easily-swappable-parameters>`,
   :doc:`bugprone-empty-catch <bugprone/empty-catch>`,
   :doc:`bugprone-exception-escape <bugprone/exception-escape>`,
   :doc:`bugprone-fold-init-type <bugprone/fold-init-type>`,
   :doc:`bugprone-forward-declaration-namespace <bugprone/forward-declaration-namespace>`,
   :doc:`bugprone-forwarding-reference-overload <bugprone/forwarding-reference-overload>`,
   :doc:`bugprone-implicit-widening-of-multiplication-result <bugprone/implicit-widening-of-multiplication-result>`, "Yes"
   :doc:`bugprone-inaccurate-erase <bugprone/inaccurate-erase>`, "Yes"
   :doc:`bugprone-inc-dec-in-conditions <bugprone/inc-dec-in-conditions>`,
   :doc:`bugprone-incorrect-enable-if <bugprone/incorrect-enable-if>`, "Yes"
   :doc:`bugprone-incorrect-roundings <bugprone/incorrect-roundings>`,
   :doc:`bugprone-infinite-loop <bugprone/infinite-loop>`,
   :doc:`bugprone-integer-division <bugprone/integer-division>`,
   :doc:`bugprone-lambda-function-name <bugprone/lambda-function-name>`,
   :doc:`bugprone-macro-parentheses <bugprone/macro-parentheses>`, "Yes"
   :doc:`bugprone-macro-repeated-side-effects <bugprone/macro-repeated-side-effects>`,
   :doc:`bugprone-misplaced-operator-in-strlen-in-alloc <bugprone/misplaced-operator-in-strlen-in-alloc>`, "Yes"
   :doc:`bugprone-misplaced-pointer-arithmetic-in-alloc <bugprone/misplaced-pointer-arithmetic-in-alloc>`, "Yes"
   :doc:`bugprone-misplaced-widening-cast <bugprone/misplaced-widening-cast>`,
   :doc:`bugprone-move-forwarding-reference <bugprone/move-forwarding-reference>`, "Yes"
   :doc:`bugprone-multi-level-implicit-pointer-conversion <bugprone/multi-level-implicit-pointer-conversion>`,
   :doc:`bugprone-multiple-new-in-one-expression <bugprone/multiple-new-in-one-expression>`,
   :doc:`bugprone-multiple-statement-macro <bugprone/multiple-statement-macro>`,
   :doc:`bugprone-no-escape <bugprone/no-escape>`,
   :doc:`bugprone-non-zero-enum-to-bool-conversion <bugprone/non-zero-enum-to-bool-conversion>`,
   :doc:`bugprone-not-null-terminated-result <bugprone/not-null-terminated-result>`, "Yes"
   :doc:`bugprone-optional-value-conversion <bugprone/optional-value-conversion>`, "Yes"
   :doc:`bugprone-parent-virtual-call <bugprone/parent-virtual-call>`, "Yes"
   :doc:`bugprone-posix-return <bugprone/posix-return>`, "Yes"
   :doc:`bugprone-redundant-branch-condition <bugprone/redundant-branch-condition>`, "Yes"
   :doc:`bugprone-reserved-identifier <bugprone/reserved-identifier>`, "Yes"
   :doc:`bugprone-shared-ptr-array-mismatch <bugprone/shared-ptr-array-mismatch>`, "Yes"
   :doc:`bugprone-signal-handler <bugprone/signal-handler>`,
   :doc:`bugprone-signed-char-misuse <bugprone/signed-char-misuse>`,
   :doc:`bugprone-sizeof-container <bugprone/sizeof-container>`,
   :doc:`bugprone-sizeof-expression <bugprone/sizeof-expression>`,
   :doc:`bugprone-spuriously-wake-up-functions <bugprone/spuriously-wake-up-functions>`,
   :doc:`bugprone-standalone-empty <bugprone/standalone-empty>`, "Yes"
   :doc:`bugprone-string-constructor <bugprone/string-constructor>`, "Yes"
   :doc:`bugprone-string-integer-assignment <bugprone/string-integer-assignment>`, "Yes"
   :doc:`bugprone-string-literal-with-embedded-nul <bugprone/string-literal-with-embedded-nul>`,
   :doc:`bugprone-stringview-nullptr <bugprone/stringview-nullptr>`, "Yes"
   :doc:`bugprone-suspicious-enum-usage <bugprone/suspicious-enum-usage>`,
   :doc:`bugprone-suspicious-include <bugprone/suspicious-include>`,
   :doc:`bugprone-suspicious-memory-comparison <bugprone/suspicious-memory-comparison>`,
   :doc:`bugprone-suspicious-memset-usage <bugprone/suspicious-memset-usage>`, "Yes"
   :doc:`bugprone-suspicious-missing-comma <bugprone/suspicious-missing-comma>`,
   :doc:`bugprone-suspicious-realloc-usage <bugprone/suspicious-realloc-usage>`,
   :doc:`bugprone-suspicious-semicolon <bugprone/suspicious-semicolon>`, "Yes"
   :doc:`bugprone-suspicious-string-compare <bugprone/suspicious-string-compare>`, "Yes"
   :doc:`bugprone-suspicious-stringview-data-usage <bugprone/suspicious-stringview-data-usage>`,
   :doc:`bugprone-swapped-arguments <bugprone/swapped-arguments>`, "Yes"
   :doc:`bugprone-switch-missing-default-case <bugprone/switch-missing-default-case>`,
   :doc:`bugprone-terminating-continue <bugprone/terminating-continue>`, "Yes"
   :doc:`bugprone-throw-keyword-missing <bugprone/throw-keyword-missing>`,
   :doc:`bugprone-too-small-loop-variable <bugprone/too-small-loop-variable>`,
   :doc:`bugprone-unchecked-optional-access <bugprone/unchecked-optional-access>`,
   :doc:`bugprone-undefined-memory-manipulation <bugprone/undefined-memory-manipulation>`,
   :doc:`bugprone-undelegated-constructor <bugprone/undelegated-constructor>`,
   :doc:`bugprone-unhandled-exception-at-new <bugprone/unhandled-exception-at-new>`,
   :doc:`bugprone-unhandled-self-assignment <bugprone/unhandled-self-assignment>`,
   :doc:`bugprone-unique-ptr-array-mismatch <bugprone/unique-ptr-array-mismatch>`, "Yes"
   :doc:`bugprone-unsafe-functions <bugprone/unsafe-functions>`,
   :doc:`bugprone-unused-local-non-trivial-variable <bugprone/unused-local-non-trivial-variable>`,
   :doc:`bugprone-unused-raii <bugprone/unused-raii>`, "Yes"
   :doc:`bugprone-unused-return-value <bugprone/unused-return-value>`,
   :doc:`bugprone-use-after-move <bugprone/use-after-move>`,
   :doc:`bugprone-virtual-near-miss <bugprone/virtual-near-miss>`, "Yes"
   :doc:`cert-dcl50-cpp <cert/dcl50-cpp>`,
   :doc:`cert-dcl58-cpp <cert/dcl58-cpp>`,
   :doc:`cert-env33-c <cert/env33-c>`,
   :doc:`cert-err33-c <cert/err33-c>`,
   :doc:`cert-err34-c <cert/err34-c>`,
   :doc:`cert-err52-cpp <cert/err52-cpp>`,
   :doc:`cert-err58-cpp <cert/err58-cpp>`,
   :doc:`cert-err60-cpp <cert/err60-cpp>`,
   :doc:`cert-flp30-c <cert/flp30-c>`,
   :doc:`cert-mem57-cpp <cert/mem57-cpp>`,
   :doc:`cert-msc50-cpp <cert/msc50-cpp>`,
   :doc:`cert-msc51-cpp <cert/msc51-cpp>`,
   :doc:`cert-oop57-cpp <cert/oop57-cpp>`,
   :doc:`cert-oop58-cpp <cert/oop58-cpp>`,
   :doc:`concurrency-mt-unsafe <concurrency/mt-unsafe>`,
   :doc:`concurrency-thread-canceltype-asynchronous <concurrency/thread-canceltype-asynchronous>`,
   :doc:`cppcoreguidelines-avoid-capturing-lambda-coroutines <cppcoreguidelines/avoid-capturing-lambda-coroutines>`,
   :doc:`cppcoreguidelines-avoid-const-or-ref-data-members <cppcoreguidelines/avoid-const-or-ref-data-members>`,
   :doc:`cppcoreguidelines-avoid-do-while <cppcoreguidelines/avoid-do-while>`,
   :doc:`cppcoreguidelines-avoid-goto <cppcoreguidelines/avoid-goto>`,
   :doc:`cppcoreguidelines-avoid-non-const-global-variables <cppcoreguidelines/avoid-non-const-global-variables>`,
   :doc:`cppcoreguidelines-avoid-reference-coroutine-parameters <cppcoreguidelines/avoid-reference-coroutine-parameters>`,
   :doc:`cppcoreguidelines-init-variables <cppcoreguidelines/init-variables>`, "Yes"
   :doc:`cppcoreguidelines-interfaces-global-init <cppcoreguidelines/interfaces-global-init>`,
   :doc:`cppcoreguidelines-macro-usage <cppcoreguidelines/macro-usage>`,
   :doc:`cppcoreguidelines-misleading-capture-default-by-value <cppcoreguidelines/misleading-capture-default-by-value>`, "Yes"
   :doc:`cppcoreguidelines-missing-std-forward <cppcoreguidelines/missing-std-forward>`,
   :doc:`cppcoreguidelines-narrowing-conversions <cppcoreguidelines/narrowing-conversions>`,
   :doc:`cppcoreguidelines-no-malloc <cppcoreguidelines/no-malloc>`,
   :doc:`cppcoreguidelines-no-suspend-with-lock <cppcoreguidelines/no-suspend-with-lock>`,
   :doc:`cppcoreguidelines-owning-memory <cppcoreguidelines/owning-memory>`,
   :doc:`cppcoreguidelines-prefer-member-initializer <cppcoreguidelines/prefer-member-initializer>`, "Yes"
   :doc:`cppcoreguidelines-pro-bounds-array-to-pointer-decay <cppcoreguidelines/pro-bounds-array-to-pointer-decay>`,
   :doc:`cppcoreguidelines-pro-bounds-constant-array-index <cppcoreguidelines/pro-bounds-constant-array-index>`, "Yes"
   :doc:`cppcoreguidelines-pro-bounds-pointer-arithmetic <cppcoreguidelines/pro-bounds-pointer-arithmetic>`,
   :doc:`cppcoreguidelines-pro-type-const-cast <cppcoreguidelines/pro-type-const-cast>`,
   :doc:`cppcoreguidelines-pro-type-cstyle-cast <cppcoreguidelines/pro-type-cstyle-cast>`, "Yes"
   :doc:`cppcoreguidelines-pro-type-member-init <cppcoreguidelines/pro-type-member-init>`, "Yes"
   :doc:`cppcoreguidelines-pro-type-reinterpret-cast <cppcoreguidelines/pro-type-reinterpret-cast>`,
   :doc:`cppcoreguidelines-pro-type-static-cast-downcast <cppcoreguidelines/pro-type-static-cast-downcast>`, "Yes"
   :doc:`cppcoreguidelines-pro-type-union-access <cppcoreguidelines/pro-type-union-access>`,
   :doc:`cppcoreguidelines-pro-type-vararg <cppcoreguidelines/pro-type-vararg>`,
   :doc:`cppcoreguidelines-rvalue-reference-param-not-moved <cppcoreguidelines/rvalue-reference-param-not-moved>`,
   :doc:`cppcoreguidelines-slicing <cppcoreguidelines/slicing>`,
   :doc:`cppcoreguidelines-special-member-functions <cppcoreguidelines/special-member-functions>`,
   :doc:`cppcoreguidelines-virtual-class-destructor <cppcoreguidelines/virtual-class-destructor>`, "Yes"
   :doc:`darwin-avoid-spinlock <darwin/avoid-spinlock>`,
   :doc:`darwin-dispatch-once-nonstatic <darwin/dispatch-once-nonstatic>`, "Yes"
   :doc:`fuchsia-default-arguments-calls <fuchsia/default-arguments-calls>`,
   :doc:`fuchsia-default-arguments-declarations <fuchsia/default-arguments-declarations>`, "Yes"
   :doc:`fuchsia-multiple-inheritance <fuchsia/multiple-inheritance>`,
   :doc:`fuchsia-overloaded-operator <fuchsia/overloaded-operator>`,
   :doc:`fuchsia-statically-constructed-objects <fuchsia/statically-constructed-objects>`,
   :doc:`fuchsia-trailing-return <fuchsia/trailing-return>`,
   :doc:`fuchsia-virtual-inheritance <fuchsia/virtual-inheritance>`,
   :doc:`google-build-explicit-make-pair <google/build-explicit-make-pair>`,
   :doc:`google-build-namespaces <google/build-namespaces>`,
   :doc:`google-build-using-namespace <google/build-using-namespace>`,
   :doc:`google-default-arguments <google/default-arguments>`,
   :doc:`google-explicit-constructor <google/explicit-constructor>`, "Yes"
   :doc:`google-global-names-in-headers <google/global-names-in-headers>`,
   :doc:`google-objc-avoid-nsobject-new <google/objc-avoid-nsobject-new>`,
   :doc:`google-objc-avoid-throwing-exception <google/objc-avoid-throwing-exception>`,
   :doc:`google-objc-function-naming <google/objc-function-naming>`,
   :doc:`google-objc-global-variable-declaration <google/objc-global-variable-declaration>`,
   :doc:`google-readability-avoid-underscore-in-googletest-name <google/readability-avoid-underscore-in-googletest-name>`,
   :doc:`google-readability-casting <google/readability-casting>`,
   :doc:`google-readability-todo <google/readability-todo>`,
   :doc:`google-runtime-int <google/runtime-int>`,
   :doc:`google-runtime-operator <google/runtime-operator>`,
   :doc:`google-upgrade-googletest-case <google/upgrade-googletest-case>`, "Yes"
   :doc:`hicpp-exception-baseclass <hicpp/exception-baseclass>`,
   :doc:`hicpp-ignored-remove-result <hicpp/ignored-remove-result>`,
   :doc:`hicpp-multiway-paths-covered <hicpp/multiway-paths-covered>`,
   :doc:`hicpp-no-assembler <hicpp/no-assembler>`,
   :doc:`hicpp-signed-bitwise <hicpp/signed-bitwise>`,
   :doc:`linuxkernel-must-use-errs <linuxkernel/must-use-errs>`,
   :doc:`llvm-header-guard <llvm/header-guard>`,
   :doc:`llvm-include-order <llvm/include-order>`, "Yes"
   :doc:`llvm-namespace-comment <llvm/namespace-comment>`,
   :doc:`llvm-prefer-isa-or-dyn-cast-in-conditionals <llvm/prefer-isa-or-dyn-cast-in-conditionals>`, "Yes"
   :doc:`llvm-prefer-register-over-unsigned <llvm/prefer-register-over-unsigned>`, "Yes"
   :doc:`llvm-twine-local <llvm/twine-local>`, "Yes"
   :doc:`llvmlibc-callee-namespace <llvmlibc/callee-namespace>`,
   :doc:`llvmlibc-implementation-in-namespace <llvmlibc/implementation-in-namespace>`,
   :doc:`llvmlibc-inline-function-decl <llvmlibc/inline-function-decl>`, "Yes"
   :doc:`llvmlibc-restrict-system-libc-headers <llvmlibc/restrict-system-libc-headers>`, "Yes"
   :doc:`misc-confusable-identifiers <misc/confusable-identifiers>`,
   :doc:`misc-const-correctness <misc/const-correctness>`, "Yes"
   :doc:`misc-coroutine-hostile-raii <misc/coroutine-hostile-raii>`,
   :doc:`misc-definitions-in-headers <misc/definitions-in-headers>`, "Yes"
   :doc:`misc-header-include-cycle <misc/header-include-cycle>`,
   :doc:`misc-include-cleaner <misc/include-cleaner>`, "Yes"
   :doc:`misc-misleading-bidirectional <misc/misleading-bidirectional>`,
   :doc:`misc-misleading-identifier <misc/misleading-identifier>`,
   :doc:`misc-misplaced-const <misc/misplaced-const>`,
   :doc:`misc-new-delete-overloads <misc/new-delete-overloads>`,
   :doc:`misc-no-recursion <misc/no-recursion>`,
   :doc:`misc-non-copyable-objects <misc/non-copyable-objects>`,
   :doc:`misc-non-private-member-variables-in-classes <misc/non-private-member-variables-in-classes>`,
   :doc:`misc-redundant-expression <misc/redundant-expression>`, "Yes"
   :doc:`misc-static-assert <misc/static-assert>`, "Yes"
   :doc:`misc-throw-by-value-catch-by-reference <misc/throw-by-value-catch-by-reference>`,
   :doc:`misc-unconventional-assign-operator <misc/unconventional-assign-operator>`,
   :doc:`misc-uniqueptr-reset-release <misc/uniqueptr-reset-release>`, "Yes"
   :doc:`misc-unused-alias-decls <misc/unused-alias-decls>`, "Yes"
   :doc:`misc-unused-parameters <misc/unused-parameters>`, "Yes"
   :doc:`misc-unused-using-decls <misc/unused-using-decls>`, "Yes"
   :doc:`misc-use-anonymous-namespace <misc/use-anonymous-namespace>`,
   :doc:`modernize-avoid-bind <modernize/avoid-bind>`, "Yes"
   :doc:`modernize-avoid-c-arrays <modernize/avoid-c-arrays>`,
   :doc:`modernize-concat-nested-namespaces <modernize/concat-nested-namespaces>`, "Yes"
   :doc:`modernize-deprecated-headers <modernize/deprecated-headers>`, "Yes"
   :doc:`modernize-deprecated-ios-base-aliases <modernize/deprecated-ios-base-aliases>`, "Yes"
   :doc:`modernize-loop-convert <modernize/loop-convert>`, "Yes"
   :doc:`modernize-macro-to-enum <modernize/macro-to-enum>`, "Yes"
   :doc:`modernize-make-shared <modernize/make-shared>`, "Yes"
   :doc:`modernize-make-unique <modernize/make-unique>`, "Yes"
   :doc:`modernize-pass-by-value <modernize/pass-by-value>`, "Yes"
   :doc:`modernize-raw-string-literal <modernize/raw-string-literal>`, "Yes"
   :doc:`modernize-redundant-void-arg <modernize/redundant-void-arg>`, "Yes"
   :doc:`modernize-replace-auto-ptr <modernize/replace-auto-ptr>`, "Yes"
   :doc:`modernize-replace-disallow-copy-and-assign-macro <modernize/replace-disallow-copy-and-assign-macro>`, "Yes"
   :doc:`modernize-replace-random-shuffle <modernize/replace-random-shuffle>`, "Yes"
   :doc:`modernize-return-braced-init-list <modernize/return-braced-init-list>`, "Yes"
   :doc:`modernize-shrink-to-fit <modernize/shrink-to-fit>`, "Yes"
   :doc:`modernize-type-traits <modernize/type-traits>`, "Yes"
   :doc:`modernize-unary-static-assert <modernize/unary-static-assert>`, "Yes"
   :doc:`modernize-use-auto <modernize/use-auto>`, "Yes"
   :doc:`modernize-use-bool-literals <modernize/use-bool-literals>`, "Yes"
   :doc:`modernize-use-constraints <modernize/use-constraints>`, "Yes"
   :doc:`modernize-use-default-member-init <modernize/use-default-member-init>`, "Yes"
   :doc:`modernize-use-designated-initializers <modernize/use-designated-initializers>`, "Yes"
   :doc:`modernize-use-emplace <modernize/use-emplace>`, "Yes"
   :doc:`modernize-use-equals-default <modernize/use-equals-default>`, "Yes"
   :doc:`modernize-use-equals-delete <modernize/use-equals-delete>`, "Yes"
   :doc:`modernize-use-nodiscard <modernize/use-nodiscard>`, "Yes"
   :doc:`modernize-use-noexcept <modernize/use-noexcept>`, "Yes"
   :doc:`modernize-use-nullptr <modernize/use-nullptr>`, "Yes"
   :doc:`modernize-use-override <modernize/use-override>`, "Yes"
   :doc:`modernize-use-starts-ends-with <modernize/use-starts-ends-with>`, "Yes"
   :doc:`modernize-use-std-numbers <modernize/use-std-numbers>`, "Yes"
   :doc:`modernize-use-std-print <modernize/use-std-print>`, "Yes"
   :doc:`modernize-use-trailing-return-type <modernize/use-trailing-return-type>`, "Yes"
   :doc:`modernize-use-transparent-functors <modernize/use-transparent-functors>`, "Yes"
   :doc:`modernize-use-uncaught-exceptions <modernize/use-uncaught-exceptions>`, "Yes"
   :doc:`modernize-use-using <modernize/use-using>`, "Yes"
   :doc:`mpi-buffer-deref <mpi/buffer-deref>`, "Yes"
   :doc:`mpi-type-mismatch <mpi/type-mismatch>`, "Yes"
   :doc:`objc-assert-equals <objc/assert-equals>`, "Yes"
   :doc:`objc-avoid-nserror-init <objc/avoid-nserror-init>`,
   :doc:`objc-dealloc-in-category <objc/dealloc-in-category>`,
   :doc:`objc-forbidden-subclassing <objc/forbidden-subclassing>`,
   :doc:`objc-missing-hash <objc/missing-hash>`,
   :doc:`objc-nsdate-formatter <objc/nsdate-formatter>`,
   :doc:`objc-nsinvocation-argument-lifetime <objc/nsinvocation-argument-lifetime>`, "Yes"
   :doc:`objc-property-declaration <objc/property-declaration>`, "Yes"
   :doc:`objc-super-self <objc/super-self>`, "Yes"
   :doc:`openmp-exception-escape <openmp/exception-escape>`,
   :doc:`openmp-use-default-none <openmp/use-default-none>`,
   :doc:`performance-avoid-endl <performance/avoid-endl>`, "Yes"
   :doc:`performance-enum-size <performance/enum-size>`,
   :doc:`performance-faster-string-find <performance/faster-string-find>`, "Yes"
   :doc:`performance-for-range-copy <performance/for-range-copy>`, "Yes"
   :doc:`performance-implicit-conversion-in-loop <performance/implicit-conversion-in-loop>`,
   :doc:`performance-inefficient-algorithm <performance/inefficient-algorithm>`, "Yes"
   :doc:`performance-inefficient-string-concatenation <performance/inefficient-string-concatenation>`,
   :doc:`performance-inefficient-vector-operation <performance/inefficient-vector-operation>`, "Yes"
   :doc:`performance-move-const-arg <performance/move-const-arg>`, "Yes"
   :doc:`performance-move-constructor-init <performance/move-constructor-init>`,
   :doc:`performance-no-automatic-move <performance/no-automatic-move>`,
   :doc:`performance-no-int-to-ptr <performance/no-int-to-ptr>`,
   :doc:`performance-noexcept-destructor <performance/noexcept-destructor>`, "Yes"
   :doc:`performance-noexcept-move-constructor <performance/noexcept-move-constructor>`, "Yes"
   :doc:`performance-noexcept-swap <performance/noexcept-swap>`, "Yes"
   :doc:`performance-trivially-destructible <performance/trivially-destructible>`, "Yes"
   :doc:`performance-type-promotion-in-math-fn <performance/type-promotion-in-math-fn>`, "Yes"
   :doc:`performance-unnecessary-copy-initialization <performance/unnecessary-copy-initialization>`, "Yes"
   :doc:`performance-unnecessary-value-param <performance/unnecessary-value-param>`, "Yes"
   :doc:`portability-restrict-system-includes <portability/restrict-system-includes>`, "Yes"
   :doc:`portability-simd-intrinsics <portability/simd-intrinsics>`,
   :doc:`portability-std-allocator-const <portability/std-allocator-const>`,
   :doc:`readability-avoid-const-params-in-decls <readability/avoid-const-params-in-decls>`, "Yes"
   :doc:`readability-avoid-nested-conditional-operator <readability/avoid-nested-conditional-operator>`,
   :doc:`readability-avoid-return-with-void-value <readability/avoid-return-with-void-value>`,
   :doc:`readability-avoid-unconditional-preprocessor-if <readability/avoid-unconditional-preprocessor-if>`,
   :doc:`readability-braces-around-statements <readability/braces-around-statements>`, "Yes"
   :doc:`readability-const-return-type <readability/const-return-type>`, "Yes"
   :doc:`readability-container-contains <readability/container-contains>`, "Yes"
   :doc:`readability-container-data-pointer <readability/container-data-pointer>`, "Yes"
   :doc:`readability-container-size-empty <readability/container-size-empty>`, "Yes"
   :doc:`readability-convert-member-functions-to-static <readability/convert-member-functions-to-static>`, "Yes"
   :doc:`readability-delete-null-pointer <readability/delete-null-pointer>`, "Yes"
   :doc:`readability-duplicate-include <readability/duplicate-include>`, "Yes"
   :doc:`readability-else-after-return <readability/else-after-return>`, "Yes"
   :doc:`readability-function-cognitive-complexity <readability/function-cognitive-complexity>`,
   :doc:`readability-function-size <readability/function-size>`,
   :doc:`readability-identifier-length <readability/identifier-length>`,
   :doc:`readability-identifier-naming <readability/identifier-naming>`, "Yes"
   :doc:`readability-implicit-bool-conversion <readability/implicit-bool-conversion>`, "Yes"
   :doc:`readability-inconsistent-declaration-parameter-name <readability/inconsistent-declaration-parameter-name>`, "Yes"
   :doc:`readability-isolate-declaration <readability/isolate-declaration>`, "Yes"
   :doc:`readability-magic-numbers <readability/magic-numbers>`,
   :doc:`readability-make-member-function-const <readability/make-member-function-const>`, "Yes"
   :doc:`readability-misleading-indentation <readability/misleading-indentation>`,
   :doc:`readability-misplaced-array-index <readability/misplaced-array-index>`, "Yes"
   :doc:`readability-named-parameter <readability/named-parameter>`, "Yes"
   :doc:`readability-non-const-parameter <readability/non-const-parameter>`, "Yes"
   :doc:`readability-operators-representation <readability/operators-representation>`, "Yes"
   :doc:`readability-qualified-auto <readability/qualified-auto>`, "Yes"
   :doc:`readability-redundant-access-specifiers <readability/redundant-access-specifiers>`, "Yes"
   :doc:`readability-redundant-casting <readability/redundant-casting>`, "Yes"
   :doc:`readability-redundant-control-flow <readability/redundant-control-flow>`, "Yes"
   :doc:`readability-redundant-declaration <readability/redundant-declaration>`, "Yes"
   :doc:`readability-redundant-function-ptr-dereference <readability/redundant-function-ptr-dereference>`, "Yes"
   :doc:`readability-redundant-inline-specifier <readability/redundant-inline-specifier>`, "Yes"
   :doc:`readability-redundant-member-init <readability/redundant-member-init>`, "Yes"
   :doc:`readability-redundant-preprocessor <readability/redundant-preprocessor>`,
   :doc:`readability-redundant-smartptr-get <readability/redundant-smartptr-get>`, "Yes"
   :doc:`readability-redundant-string-cstr <readability/redundant-string-cstr>`, "Yes"
   :doc:`readability-redundant-string-init <readability/redundant-string-init>`, "Yes"
   :doc:`readability-reference-to-constructed-temporary <readability/reference-to-constructed-temporary>`,
   :doc:`readability-simplify-boolean-expr <readability/simplify-boolean-expr>`, "Yes"
   :doc:`readability-simplify-subscript-expr <readability/simplify-subscript-expr>`, "Yes"
   :doc:`readability-static-accessed-through-instance <readability/static-accessed-through-instance>`, "Yes"
   :doc:`readability-static-definition-in-anonymous-namespace <readability/static-definition-in-anonymous-namespace>`, "Yes"
   :doc:`readability-string-compare <readability/string-compare>`, "Yes"
   :doc:`readability-suspicious-call-argument <readability/suspicious-call-argument>`,
   :doc:`readability-uniqueptr-delete-release <readability/uniqueptr-delete-release>`, "Yes"
   :doc:`readability-uppercase-literal-suffix <readability/uppercase-literal-suffix>`, "Yes"
   :doc:`readability-use-anyofallof <readability/use-anyofallof>`,
   :doc:`readability-use-std-min-max <readability/use-std-min-max>`, "Yes"
   :doc:`zircon-temporary-objects <zircon/temporary-objects>`,


.. csv-table:: Aliases..
   :header: "Name", "Redirect", "Offers fixes"

   :doc:`bugprone-narrowing-conversions <bugprone/narrowing-conversions>`, :doc:`cppcoreguidelines-narrowing-conversions <cppcoreguidelines/narrowing-conversions>`,
   :doc:`cert-con36-c <cert/con36-c>`, :doc:`bugprone-spuriously-wake-up-functions <bugprone/spuriously-wake-up-functions>`,
   :doc:`cert-con54-cpp <cert/con54-cpp>`, :doc:`bugprone-spuriously-wake-up-functions <bugprone/spuriously-wake-up-functions>`,
   :doc:`cert-dcl03-c <cert/dcl03-c>`, :doc:`misc-static-assert <misc/static-assert>`, "Yes"
   :doc:`cert-dcl16-c <cert/dcl16-c>`, :doc:`readability-uppercase-literal-suffix <readability/uppercase-literal-suffix>`, "Yes"
   :doc:`cert-dcl37-c <cert/dcl37-c>`, :doc:`bugprone-reserved-identifier <bugprone/reserved-identifier>`, "Yes"
   :doc:`cert-dcl51-cpp <cert/dcl51-cpp>`, :doc:`bugprone-reserved-identifier <bugprone/reserved-identifier>`, "Yes"
   :doc:`cert-dcl54-cpp <cert/dcl54-cpp>`, :doc:`misc-new-delete-overloads <misc/new-delete-overloads>`,
   :doc:`cert-dcl59-cpp <cert/dcl59-cpp>`, :doc:`google-build-namespaces <google/build-namespaces>`,
   :doc:`cert-err09-cpp <cert/err09-cpp>`, :doc:`misc-throw-by-value-catch-by-reference <misc/throw-by-value-catch-by-reference>`,
   :doc:`cert-err61-cpp <cert/err61-cpp>`, :doc:`misc-throw-by-value-catch-by-reference <misc/throw-by-value-catch-by-reference>`,
   :doc:`cert-exp42-c <cert/exp42-c>`, :doc:`bugprone-suspicious-memory-comparison <bugprone/suspicious-memory-comparison>`,
   :doc:`cert-fio38-c <cert/fio38-c>`, :doc:`misc-non-copyable-objects <misc/non-copyable-objects>`,
   :doc:`cert-flp37-c <cert/flp37-c>`, :doc:`bugprone-suspicious-memory-comparison <bugprone/suspicious-memory-comparison>`,
   :doc:`cert-msc24-c <cert/msc24-c>`, :doc:`bugprone-unsafe-functions <bugprone/unsafe-functions>`,
   :doc:`cert-msc30-c <cert/msc30-c>`, :doc:`cert-msc50-cpp <cert/msc50-cpp>`,
   :doc:`cert-msc32-c <cert/msc32-c>`, :doc:`cert-msc51-cpp <cert/msc51-cpp>`,
   :doc:`cert-msc33-c <cert/msc33-c>`, :doc:`bugprone-unsafe-functions <bugprone/unsafe-functions>`,
   :doc:`cert-msc54-cpp <cert/msc54-cpp>`, :doc:`bugprone-signal-handler <bugprone/signal-handler>`,
   :doc:`cert-oop11-cpp <cert/oop11-cpp>`, :doc:`performance-move-constructor-init <performance/move-constructor-init>`,
   :doc:`cert-oop54-cpp <cert/oop54-cpp>`, :doc:`bugprone-unhandled-self-assignment <bugprone/unhandled-self-assignment>`,
   :doc:`cert-pos44-c <cert/pos44-c>`, :doc:`bugprone-bad-signal-to-kill-thread <bugprone/bad-signal-to-kill-thread>`,
   :doc:`cert-pos47-c <cert/pos47-c>`, :doc:`concurrency-thread-canceltype-asynchronous <concurrency/thread-canceltype-asynchronous>`,
   :doc:`cert-sig30-c <cert/sig30-c>`, :doc:`bugprone-signal-handler <bugprone/signal-handler>`,
   :doc:`cert-str34-c <cert/str34-c>`, :doc:`bugprone-signed-char-misuse <bugprone/signed-char-misuse>`,
   :doc:`clang-analyzer-core.BitwiseShift <clang-analyzer/core.BitwiseShift>`, `Clang Static Analyzer core.BitwiseShift <https://clang.llvm.org/docs/analyzer/checkers.html#core-bitwiseshift>`_,
   :doc:`clang-analyzer-core.CallAndMessage <clang-analyzer/core.CallAndMessage>`, `Clang Static Analyzer core.CallAndMessage <https://clang.llvm.org/docs/analyzer/checkers.html#core-callandmessage>`_,
   :doc:`clang-analyzer-core.DivideZero <clang-analyzer/core.DivideZero>`, `Clang Static Analyzer core.DivideZero <https://clang.llvm.org/docs/analyzer/checkers.html#core-dividezero>`_,
   :doc:`clang-analyzer-core.NonNullParamChecker <clang-analyzer/core.NonNullParamChecker>`, `Clang Static Analyzer core.NonNullParamChecker <https://clang.llvm.org/docs/analyzer/checkers.html#core-nonnullparamchecker>`_,
   :doc:`clang-analyzer-core.NullDereference <clang-analyzer/core.NullDereference>`, `Clang Static Analyzer core.NullDereference <https://clang.llvm.org/docs/analyzer/checkers.html#core-nulldereference>`_,
   :doc:`clang-analyzer-core.StackAddressEscape <clang-analyzer/core.StackAddressEscape>`, `Clang Static Analyzer core.StackAddressEscape <https://clang.llvm.org/docs/analyzer/checkers.html#core-stackaddressescape>`_,
   :doc:`clang-analyzer-core.UndefinedBinaryOperatorResult <clang-analyzer/core.UndefinedBinaryOperatorResult>`, `Clang Static Analyzer core.UndefinedBinaryOperatorResult <https://clang.llvm.org/docs/analyzer/checkers.html#core-undefinedbinaryoperatorresult>`_,
   :doc:`clang-analyzer-core.VLASize <clang-analyzer/core.VLASize>`, `Clang Static Analyzer core.VLASize <https://clang.llvm.org/docs/analyzer/checkers.html#core-vlasize>`_,
   :doc:`clang-analyzer-core.uninitialized.ArraySubscript <clang-analyzer/core.uninitialized.ArraySubscript>`, `Clang Static Analyzer core.uninitialized.ArraySubscript <https://clang.llvm.org/docs/analyzer/checkers.html#core-uninitialized-arraysubscript>`_,
   :doc:`clang-analyzer-core.uninitialized.Assign <clang-analyzer/core.uninitialized.Assign>`, `Clang Static Analyzer core.uninitialized.Assign <https://clang.llvm.org/docs/analyzer/checkers.html#core-uninitialized-assign>`_,
   :doc:`clang-analyzer-core.uninitialized.Branch <clang-analyzer/core.uninitialized.Branch>`, `Clang Static Analyzer core.uninitialized.Branch <https://clang.llvm.org/docs/analyzer/checkers.html#core-uninitialized-branch>`_,
   :doc:`clang-analyzer-core.uninitialized.CapturedBlockVariable <clang-analyzer/core.uninitialized.CapturedBlockVariable>`, `Clang Static Analyzer core.uninitialized.CapturedBlockVariable <https://clang.llvm.org/docs/analyzer/checkers.html#core-uninitialized-capturedblockvariable>`_,
   :doc:`clang-analyzer-core.uninitialized.NewArraySize <clang-analyzer/core.uninitialized.NewArraySize>`, `Clang Static Analyzer core.uninitialized.NewArraySize <https://clang.llvm.org/docs/analyzer/checkers.html#core-uninitialized-newarraysize>`_,
   :doc:`clang-analyzer-core.uninitialized.UndefReturn <clang-analyzer/core.uninitialized.UndefReturn>`, `Clang Static Analyzer core.uninitialized.UndefReturn <https://clang.llvm.org/docs/analyzer/checkers.html#core-uninitialized-undefreturn>`_,
   :doc:`clang-analyzer-cplusplus.InnerPointer <clang-analyzer/cplusplus.InnerPointer>`, `Clang Static Analyzer cplusplus.InnerPointer <https://clang.llvm.org/docs/analyzer/checkers.html#cplusplus-innerpointer>`_,
   :doc:`clang-analyzer-cplusplus.Move <clang-analyzer/cplusplus.Move>`, Clang Static Analyzer cplusplus.Move,
   :doc:`clang-analyzer-cplusplus.NewDelete <clang-analyzer/cplusplus.NewDelete>`, `Clang Static Analyzer cplusplus.NewDelete <https://clang.llvm.org/docs/analyzer/checkers.html#cplusplus-newdelete>`_,
   :doc:`clang-analyzer-cplusplus.NewDeleteLeaks <clang-analyzer/cplusplus.NewDeleteLeaks>`, `Clang Static Analyzer cplusplus.NewDeleteLeaks <https://clang.llvm.org/docs/analyzer/checkers.html#cplusplus-newdeleteleaks>`_,
   :doc:`clang-analyzer-cplusplus.PlacementNew <clang-analyzer/cplusplus.PlacementNew>`, `Clang Static Analyzer cplusplus.PlacementNew <https://clang.llvm.org/docs/analyzer/checkers.html#cplusplus-placementnew>`_,
   :doc:`clang-analyzer-cplusplus.PureVirtualCall <clang-analyzer/cplusplus.PureVirtualCall>`, Clang Static Analyzer cplusplus.PureVirtualCall,
   :doc:`clang-analyzer-cplusplus.StringChecker <clang-analyzer/cplusplus.StringChecker>`, `Clang Static Analyzer cplusplus.StringChecker <https://clang.llvm.org/docs/analyzer/checkers.html#cplusplus-stringchecker>`_,
   :doc:`clang-analyzer-deadcode.DeadStores <clang-analyzer/deadcode.DeadStores>`, `Clang Static Analyzer deadcode.DeadStores <https://clang.llvm.org/docs/analyzer/checkers.html#deadcode-deadstores>`_,
   :doc:`clang-analyzer-fuchsia.HandleChecker <clang-analyzer/fuchsia.HandleChecker>`, `Clang Static Analyzer fuchsia.HandleChecker <https://clang.llvm.org/docs/analyzer/checkers.html#fuchsia-handlechecker>`_,
   :doc:`clang-analyzer-nullability.NullPassedToNonnull <clang-analyzer/nullability.NullPassedToNonnull>`, `Clang Static Analyzer nullability.NullPassedToNonnull <https://clang.llvm.org/docs/analyzer/checkers.html#nullability-nullpassedtononnull>`_,
   :doc:`clang-analyzer-nullability.NullReturnedFromNonnull <clang-analyzer/nullability.NullReturnedFromNonnull>`, `Clang Static Analyzer nullability.NullReturnedFromNonnull <https://clang.llvm.org/docs/analyzer/checkers.html#nullability-nullreturnedfromnonnull>`_,
   :doc:`clang-analyzer-nullability.NullableDereferenced <clang-analyzer/nullability.NullableDereferenced>`, `Clang Static Analyzer nullability.NullableDereferenced <https://clang.llvm.org/docs/analyzer/checkers.html#nullability-nullabledereferenced>`_,
   :doc:`clang-analyzer-nullability.NullablePassedToNonnull <clang-analyzer/nullability.NullablePassedToNonnull>`, `Clang Static Analyzer nullability.NullablePassedToNonnull <https://clang.llvm.org/docs/analyzer/checkers.html#nullability-nullablepassedtononnull>`_,
   :doc:`clang-analyzer-nullability.NullableReturnedFromNonnull <clang-analyzer/nullability.NullableReturnedFromNonnull>`, `Clang Static Analyzer nullability.NullableReturnedFromNonnull <https://clang.llvm.org/docs/analyzer/checkers.html#nullability-nullablereturnedfromnonnull>`_,
   :doc:`clang-analyzer-optin.core.EnumCastOutOfRange <clang-analyzer/optin.core.EnumCastOutOfRange>`, `Clang Static Analyzer optin.core.EnumCastOutOfRange <https://clang.llvm.org/docs/analyzer/checkers.html#optin-core-enumcastoutofrange>`_,
   :doc:`clang-analyzer-optin.cplusplus.UninitializedObject <clang-analyzer/optin.cplusplus.UninitializedObject>`, `Clang Static Analyzer optin.cplusplus.UninitializedObject <https://clang.llvm.org/docs/analyzer/checkers.html#optin-cplusplus-uninitializedobject>`_,
   :doc:`clang-analyzer-optin.cplusplus.VirtualCall <clang-analyzer/optin.cplusplus.VirtualCall>`, `Clang Static Analyzer optin.cplusplus.VirtualCall <https://clang.llvm.org/docs/analyzer/checkers.html#optin-cplusplus-virtualcall>`_,
   :doc:`clang-analyzer-optin.mpi.MPI-Checker <clang-analyzer/optin.mpi.MPI-Checker>`, `Clang Static Analyzer optin.mpi.MPI-Checker <https://clang.llvm.org/docs/analyzer/checkers.html#optin-mpi-mpi-checker>`_,
   :doc:`clang-analyzer-optin.osx.OSObjectCStyleCast <clang-analyzer/optin.osx.OSObjectCStyleCast>`, Clang Static Analyzer optin.osx.OSObjectCStyleCast,
   :doc:`clang-analyzer-optin.osx.cocoa.localizability.EmptyLocalizationContextChecker <clang-analyzer/optin.osx.cocoa.localizability.EmptyLocalizationContextChecker>`, `Clang Static Analyzer optin.osx.cocoa.localizability.EmptyLocalizationContextChecker <https://clang.llvm.org/docs/analyzer/checkers.html#optin-osx-cocoa-localizability-emptylocalizationcontextchecker>`_,
   :doc:`clang-analyzer-optin.osx.cocoa.localizability.NonLocalizedStringChecker <clang-analyzer/optin.osx.cocoa.localizability.NonLocalizedStringChecker>`, `Clang Static Analyzer optin.osx.cocoa.localizability.NonLocalizedStringChecker <https://clang.llvm.org/docs/analyzer/checkers.html#optin-osx-cocoa-localizability-nonlocalizedstringchecker>`_,
   :doc:`clang-analyzer-optin.performance.GCDAntipattern <clang-analyzer/optin.performance.GCDAntipattern>`, `Clang Static Analyzer optin.performance.GCDAntipattern <https://clang.llvm.org/docs/analyzer/checkers.html#optin-performance-gcdantipattern>`_,
   :doc:`clang-analyzer-optin.performance.Padding <clang-analyzer/optin.performance.Padding>`, `Clang Static Analyzer optin.performance.Padding <https://clang.llvm.org/docs/analyzer/checkers.html#optin-performance-padding>`_,
   :doc:`clang-analyzer-optin.portability.UnixAPI <clang-analyzer/optin.portability.UnixAPI>`, `Clang Static Analyzer optin.portability.UnixAPI <https://clang.llvm.org/docs/analyzer/checkers.html#optin-portability-unixapi>`_,
   :doc:`clang-analyzer-osx.API <clang-analyzer/osx.API>`, `Clang Static Analyzer osx.API <https://clang.llvm.org/docs/analyzer/checkers.html#osx-api>`_,
   :doc:`clang-analyzer-osx.MIG <clang-analyzer/osx.MIG>`, Clang Static Analyzer osx.MIG,
   :doc:`clang-analyzer-osx.NumberObjectConversion <clang-analyzer/osx.NumberObjectConversion>`, `Clang Static Analyzer osx.NumberObjectConversion <https://clang.llvm.org/docs/analyzer/checkers.html#osx-numberobjectconversion>`_,
   :doc:`clang-analyzer-osx.OSObjectRetainCount <clang-analyzer/osx.OSObjectRetainCount>`, Clang Static Analyzer osx.OSObjectRetainCount,
   :doc:`clang-analyzer-osx.ObjCProperty <clang-analyzer/osx.ObjCProperty>`, `Clang Static Analyzer osx.ObjCProperty <https://clang.llvm.org/docs/analyzer/checkers.html#osx-objcproperty>`_,
   :doc:`clang-analyzer-osx.SecKeychainAPI <clang-analyzer/osx.SecKeychainAPI>`, `Clang Static Analyzer osx.SecKeychainAPI <https://clang.llvm.org/docs/analyzer/checkers.html#osx-seckeychainapi>`_,
   :doc:`clang-analyzer-osx.cocoa.AtSync <clang-analyzer/osx.cocoa.AtSync>`, `Clang Static Analyzer osx.cocoa.AtSync <https://clang.llvm.org/docs/analyzer/checkers.html#osx-cocoa-atsync>`_,
   :doc:`clang-analyzer-osx.cocoa.AutoreleaseWrite <clang-analyzer/osx.cocoa.AutoreleaseWrite>`, `Clang Static Analyzer osx.cocoa.AutoreleaseWrite <https://clang.llvm.org/docs/analyzer/checkers.html#osx-cocoa-autoreleasewrite>`_,
   :doc:`clang-analyzer-osx.cocoa.ClassRelease <clang-analyzer/osx.cocoa.ClassRelease>`, `Clang Static Analyzer osx.cocoa.ClassRelease <https://clang.llvm.org/docs/analyzer/checkers.html#osx-cocoa-classrelease>`_,
   :doc:`clang-analyzer-osx.cocoa.Dealloc <clang-analyzer/osx.cocoa.Dealloc>`, `Clang Static Analyzer osx.cocoa.Dealloc <https://clang.llvm.org/docs/analyzer/checkers.html#osx-cocoa-dealloc>`_,
   :doc:`clang-analyzer-osx.cocoa.IncompatibleMethodTypes <clang-analyzer/osx.cocoa.IncompatibleMethodTypes>`, `Clang Static Analyzer osx.cocoa.IncompatibleMethodTypes <https://clang.llvm.org/docs/analyzer/checkers.html#osx-cocoa-incompatiblemethodtypes>`_,
   :doc:`clang-analyzer-osx.cocoa.Loops <clang-analyzer/osx.cocoa.Loops>`, `Clang Static Analyzer osx.cocoa.Loops <https://clang.llvm.org/docs/analyzer/checkers.html#osx-cocoa-loops>`_,
   :doc:`clang-analyzer-osx.cocoa.MissingSuperCall <clang-analyzer/osx.cocoa.MissingSuperCall>`, `Clang Static Analyzer osx.cocoa.MissingSuperCall <https://clang.llvm.org/docs/analyzer/checkers.html#osx-cocoa-missingsupercall>`_,
   :doc:`clang-analyzer-osx.cocoa.NSAutoreleasePool <clang-analyzer/osx.cocoa.NSAutoreleasePool>`, `Clang Static Analyzer osx.cocoa.NSAutoreleasePool <https://clang.llvm.org/docs/analyzer/checkers.html#osx-cocoa-nsautoreleasepool>`_,
   :doc:`clang-analyzer-osx.cocoa.NSError <clang-analyzer/osx.cocoa.NSError>`, `Clang Static Analyzer osx.cocoa.NSError <https://clang.llvm.org/docs/analyzer/checkers.html#osx-cocoa-nserror>`_,
   :doc:`clang-analyzer-osx.cocoa.NilArg <clang-analyzer/osx.cocoa.NilArg>`, `Clang Static Analyzer osx.cocoa.NilArg <https://clang.llvm.org/docs/analyzer/checkers.html#osx-cocoa-nilarg>`_,
   :doc:`clang-analyzer-osx.cocoa.NonNilReturnValue <clang-analyzer/osx.cocoa.NonNilReturnValue>`, `Clang Static Analyzer osx.cocoa.NonNilReturnValue <https://clang.llvm.org/docs/analyzer/checkers.html#osx-cocoa-nonnilreturnvalue>`_,
   :doc:`clang-analyzer-osx.cocoa.ObjCGenerics <clang-analyzer/osx.cocoa.ObjCGenerics>`, `Clang Static Analyzer osx.cocoa.ObjCGenerics <https://clang.llvm.org/docs/analyzer/checkers.html#osx-cocoa-objcgenerics>`_,
   :doc:`clang-analyzer-osx.cocoa.RetainCount <clang-analyzer/osx.cocoa.RetainCount>`, `Clang Static Analyzer osx.cocoa.RetainCount <https://clang.llvm.org/docs/analyzer/checkers.html#osx-cocoa-retaincount>`_,
   :doc:`clang-analyzer-osx.cocoa.RunLoopAutoreleaseLeak <clang-analyzer/osx.cocoa.RunLoopAutoreleaseLeak>`, `Clang Static Analyzer osx.cocoa.RunLoopAutoreleaseLeak <https://clang.llvm.org/docs/analyzer/checkers.html#osx-cocoa-runloopautoreleaseleak>`_,
   :doc:`clang-analyzer-osx.cocoa.SelfInit <clang-analyzer/osx.cocoa.SelfInit>`, `Clang Static Analyzer osx.cocoa.SelfInit <https://clang.llvm.org/docs/analyzer/checkers.html#osx-cocoa-selfinit>`_,
   :doc:`clang-analyzer-osx.cocoa.SuperDealloc <clang-analyzer/osx.cocoa.SuperDealloc>`, `Clang Static Analyzer osx.cocoa.SuperDealloc <https://clang.llvm.org/docs/analyzer/checkers.html#osx-cocoa-superdealloc>`_,
   :doc:`clang-analyzer-osx.cocoa.UnusedIvars <clang-analyzer/osx.cocoa.UnusedIvars>`, `Clang Static Analyzer osx.cocoa.UnusedIvars <https://clang.llvm.org/docs/analyzer/checkers.html#osx-cocoa-unusedivars>`_,
   :doc:`clang-analyzer-osx.cocoa.VariadicMethodTypes <clang-analyzer/osx.cocoa.VariadicMethodTypes>`, `Clang Static Analyzer osx.cocoa.VariadicMethodTypes <https://clang.llvm.org/docs/analyzer/checkers.html#osx-cocoa-variadicmethodtypes>`_,
   :doc:`clang-analyzer-osx.coreFoundation.CFError <clang-analyzer/osx.coreFoundation.CFError>`, `Clang Static Analyzer osx.coreFoundation.CFError <https://clang.llvm.org/docs/analyzer/checkers.html#osx-corefoundation-cferror>`_,
   :doc:`clang-analyzer-osx.coreFoundation.CFNumber <clang-analyzer/osx.coreFoundation.CFNumber>`, `Clang Static Analyzer osx.coreFoundation.CFNumber <https://clang.llvm.org/docs/analyzer/checkers.html#osx-corefoundation-cfnumber>`_,
   :doc:`clang-analyzer-osx.coreFoundation.CFRetainRelease <clang-analyzer/osx.coreFoundation.CFRetainRelease>`, `Clang Static Analyzer osx.coreFoundation.CFRetainRelease <https://clang.llvm.org/docs/analyzer/checkers.html#osx-corefoundation-cfretainrelease>`_,
   :doc:`clang-analyzer-osx.coreFoundation.containers.OutOfBounds <clang-analyzer/osx.coreFoundation.containers.OutOfBounds>`, `Clang Static Analyzer osx.coreFoundation.containers.OutOfBounds <https://clang.llvm.org/docs/analyzer/checkers.html#osx-corefoundation-containers-outofbounds>`_,
   :doc:`clang-analyzer-osx.coreFoundation.containers.PointerSizedValues <clang-analyzer/osx.coreFoundation.containers.PointerSizedValues>`, `Clang Static Analyzer osx.coreFoundation.containers.PointerSizedValues <https://clang.llvm.org/docs/analyzer/checkers.html#osx-corefoundation-containers-pointersizedvalues>`_,
   :doc:`clang-analyzer-security.FloatLoopCounter <clang-analyzer/security.FloatLoopCounter>`, `Clang Static Analyzer security.FloatLoopCounter <https://clang.llvm.org/docs/analyzer/checkers.html#security-floatloopcounter>`_,
   :doc:`clang-analyzer-security.cert.env.InvalidPtr <clang-analyzer/security.cert.env.InvalidPtr>`, `Clang Static Analyzer security.cert.env.InvalidPtr <https://clang.llvm.org/docs/analyzer/checkers.html#security-cert-env-invalidptr>`_,
   :doc:`clang-analyzer-security.insecureAPI.DeprecatedOrUnsafeBufferHandling <clang-analyzer/security.insecureAPI.DeprecatedOrUnsafeBufferHandling>`, `Clang Static Analyzer security.insecureAPI.DeprecatedOrUnsafeBufferHandling <https://clang.llvm.org/docs/analyzer/checkers.html#security-insecureapi-deprecatedorunsafebufferhandling>`_,
   :doc:`clang-analyzer-security.insecureAPI.UncheckedReturn <clang-analyzer/security.insecureAPI.UncheckedReturn>`, `Clang Static Analyzer security.insecureAPI.UncheckedReturn <https://clang.llvm.org/docs/analyzer/checkers.html#security-insecureapi-uncheckedreturn>`_,
   :doc:`clang-analyzer-security.insecureAPI.bcmp <clang-analyzer/security.insecureAPI.bcmp>`, `Clang Static Analyzer security.insecureAPI.bcmp <https://clang.llvm.org/docs/analyzer/checkers.html#security-insecureapi-bcmp>`_,
   :doc:`clang-analyzer-security.insecureAPI.bcopy <clang-analyzer/security.insecureAPI.bcopy>`, `Clang Static Analyzer security.insecureAPI.bcopy <https://clang.llvm.org/docs/analyzer/checkers.html#security-insecureapi-bcopy>`_,
   :doc:`clang-analyzer-security.insecureAPI.bzero <clang-analyzer/security.insecureAPI.bzero>`, `Clang Static Analyzer security.insecureAPI.bzero <https://clang.llvm.org/docs/analyzer/checkers.html#security-insecureapi-bzero>`_,
   :doc:`clang-analyzer-security.insecureAPI.decodeValueOfObjCType <clang-analyzer/security.insecureAPI.decodeValueOfObjCType>`, Clang Static Analyzer security.insecureAPI.decodeValueOfObjCType,
   :doc:`clang-analyzer-security.insecureAPI.getpw <clang-analyzer/security.insecureAPI.getpw>`, `Clang Static Analyzer security.insecureAPI.getpw <https://clang.llvm.org/docs/analyzer/checkers.html#security-insecureapi-getpw>`_,
   :doc:`clang-analyzer-security.insecureAPI.gets <clang-analyzer/security.insecureAPI.gets>`, `Clang Static Analyzer security.insecureAPI.gets <https://clang.llvm.org/docs/analyzer/checkers.html#security-insecureapi-gets>`_,
   :doc:`clang-analyzer-security.insecureAPI.mkstemp <clang-analyzer/security.insecureAPI.mkstemp>`, `Clang Static Analyzer security.insecureAPI.mkstemp <https://clang.llvm.org/docs/analyzer/checkers.html#security-insecureapi-mkstemp>`_,
   :doc:`clang-analyzer-security.insecureAPI.mktemp <clang-analyzer/security.insecureAPI.mktemp>`, `Clang Static Analyzer security.insecureAPI.mktemp <https://clang.llvm.org/docs/analyzer/checkers.html#security-insecureapi-mktemp>`_,
   :doc:`clang-analyzer-security.insecureAPI.rand <clang-analyzer/security.insecureAPI.rand>`, `Clang Static Analyzer security.insecureAPI.rand <https://clang.llvm.org/docs/analyzer/checkers.html#security-insecureapi-rand>`_,
   :doc:`clang-analyzer-security.insecureAPI.strcpy <clang-analyzer/security.insecureAPI.strcpy>`, `Clang Static Analyzer security.insecureAPI.strcpy <https://clang.llvm.org/docs/analyzer/checkers.html#security-insecureapi-strcpy>`_,
   :doc:`clang-analyzer-security.insecureAPI.vfork <clang-analyzer/security.insecureAPI.vfork>`, `Clang Static Analyzer security.insecureAPI.vfork <https://clang.llvm.org/docs/analyzer/checkers.html#security-insecureapi-vfork>`_,
   :doc:`clang-analyzer-unix.API <clang-analyzer/unix.API>`, `Clang Static Analyzer unix.API <https://clang.llvm.org/docs/analyzer/checkers.html#unix-api>`_,
   :doc:`clang-analyzer-unix.Errno <clang-analyzer/unix.Errno>`, `Clang Static Analyzer unix.Errno <https://clang.llvm.org/docs/analyzer/checkers.html#unix-errno>`_,
   :doc:`clang-analyzer-unix.Malloc <clang-analyzer/unix.Malloc>`, `Clang Static Analyzer unix.Malloc <https://clang.llvm.org/docs/analyzer/checkers.html#unix-malloc>`_,
   :doc:`clang-analyzer-unix.MallocSizeof <clang-analyzer/unix.MallocSizeof>`, `Clang Static Analyzer unix.MallocSizeof <https://clang.llvm.org/docs/analyzer/checkers.html#unix-mallocsizeof>`_,
   :doc:`clang-analyzer-unix.MismatchedDeallocator <clang-analyzer/unix.MismatchedDeallocator>`, `Clang Static Analyzer unix.MismatchedDeallocator <https://clang.llvm.org/docs/analyzer/checkers.html#unix-mismatcheddeallocator>`_,
   :doc:`clang-analyzer-unix.StdCLibraryFunctions <clang-analyzer/unix.StdCLibraryFunctions>`, `Clang Static Analyzer unix.StdCLibraryFunctions <https://clang.llvm.org/docs/analyzer/checkers.html#unix-stdclibraryfunctions>`_,
   :doc:`clang-analyzer-unix.Vfork <clang-analyzer/unix.Vfork>`, `Clang Static Analyzer unix.Vfork <https://clang.llvm.org/docs/analyzer/checkers.html#unix-vfork>`_,
   :doc:`clang-analyzer-unix.cstring.BadSizeArg <clang-analyzer/unix.cstring.BadSizeArg>`, `Clang Static Analyzer unix.cstring.BadSizeArg <https://clang.llvm.org/docs/analyzer/checkers.html#unix-cstring-badsizearg>`_,
   :doc:`clang-analyzer-unix.cstring.NullArg <clang-analyzer/unix.cstring.NullArg>`, `Clang Static Analyzer unix.cstring.NullArg <https://clang.llvm.org/docs/analyzer/checkers.html#unix-cstring-nullarg>`_,
   :doc:`clang-analyzer-valist.CopyToSelf <clang-analyzer/valist.CopyToSelf>`, Clang Static Analyzer valist.CopyToSelf,
   :doc:`clang-analyzer-valist.Uninitialized <clang-analyzer/valist.Uninitialized>`, Clang Static Analyzer valist.Uninitialized,
   :doc:`clang-analyzer-valist.Unterminated <clang-analyzer/valist.Unterminated>`, Clang Static Analyzer valist.Unterminated,
   :doc:`clang-analyzer-webkit.NoUncountedMemberChecker <clang-analyzer/webkit.NoUncountedMemberChecker>`, `Clang Static Analyzer webkit.NoUncountedMemberChecker <https://clang.llvm.org/docs/analyzer/checkers.html#webkit-nouncountedmemberchecker>`_,
   :doc:`clang-analyzer-webkit.RefCntblBaseVirtualDtor <clang-analyzer/webkit.RefCntblBaseVirtualDtor>`, `Clang Static Analyzer webkit.RefCntblBaseVirtualDtor <https://clang.llvm.org/docs/analyzer/checkers.html#webkit-refcntblbasevirtualdtor>`_,
   :doc:`clang-analyzer-webkit.UncountedLambdaCapturesChecker <clang-analyzer/webkit.UncountedLambdaCapturesChecker>`, `Clang Static Analyzer webkit.UncountedLambdaCapturesChecker <https://clang.llvm.org/docs/analyzer/checkers.html#webkit-uncountedlambdacaptureschecker>`_,
   :doc:`cppcoreguidelines-avoid-c-arrays <cppcoreguidelines/avoid-c-arrays>`, :doc:`modernize-avoid-c-arrays <modernize/avoid-c-arrays>`,
   :doc:`cppcoreguidelines-avoid-magic-numbers <cppcoreguidelines/avoid-magic-numbers>`, :doc:`readability-magic-numbers <readability/magic-numbers>`,
   :doc:`cppcoreguidelines-c-copy-assignment-signature <cppcoreguidelines/c-copy-assignment-signature>`, :doc:`misc-unconventional-assign-operator <misc/unconventional-assign-operator>`,
   :doc:`cppcoreguidelines-explicit-virtual-functions <cppcoreguidelines/explicit-virtual-functions>`, :doc:`modernize-use-override <modernize/use-override>`, "Yes"
   :doc:`cppcoreguidelines-macro-to-enum <cppcoreguidelines/macro-to-enum>`, :doc:`modernize-macro-to-enum <modernize/macro-to-enum>`, "Yes"
   :doc:`cppcoreguidelines-noexcept-destructor <cppcoreguidelines/noexcept-destructor>`, :doc:`performance-noexcept-destructor <performance/noexcept-destructor>`, "Yes"
   :doc:`cppcoreguidelines-noexcept-move-operations <cppcoreguidelines/noexcept-move-operations>`, :doc:`performance-noexcept-move-constructor <performance/noexcept-move-constructor>`, "Yes"
   :doc:`cppcoreguidelines-noexcept-swap <cppcoreguidelines/noexcept-swap>`, :doc:`performance-noexcept-swap <performance/noexcept-swap>`, "Yes"
   :doc:`cppcoreguidelines-non-private-member-variables-in-classes <cppcoreguidelines/non-private-member-variables-in-classes>`, :doc:`misc-non-private-member-variables-in-classes <misc/non-private-member-variables-in-classes>`,
   :doc:`cppcoreguidelines-use-default-member-init <cppcoreguidelines/use-default-member-init>`, :doc:`modernize-use-default-member-init <modernize/use-default-member-init>`, "Yes"
   :doc:`fuchsia-header-anon-namespaces <fuchsia/header-anon-namespaces>`, :doc:`google-build-namespaces <google/build-namespaces>`,
   :doc:`google-readability-braces-around-statements <google/readability-braces-around-statements>`, :doc:`readability-braces-around-statements <readability/braces-around-statements>`, "Yes"
   :doc:`google-readability-function-size <google/readability-function-size>`, :doc:`readability-function-size <readability/function-size>`,
   :doc:`google-readability-namespace-comments <google/readability-namespace-comments>`, :doc:`llvm-namespace-comment <llvm/namespace-comment>`,
   :doc:`hicpp-avoid-c-arrays <hicpp/avoid-c-arrays>`, :doc:`modernize-avoid-c-arrays <modernize/avoid-c-arrays>`,
   :doc:`hicpp-avoid-goto <hicpp/avoid-goto>`, :doc:`cppcoreguidelines-avoid-goto <cppcoreguidelines/avoid-goto>`,
   :doc:`hicpp-braces-around-statements <hicpp/braces-around-statements>`, :doc:`readability-braces-around-statements <readability/braces-around-statements>`, "Yes"
   :doc:`hicpp-deprecated-headers <hicpp/deprecated-headers>`, :doc:`modernize-deprecated-headers <modernize/deprecated-headers>`, "Yes"
   :doc:`hicpp-explicit-conversions <hicpp/explicit-conversions>`, :doc:`google-explicit-constructor <google/explicit-constructor>`, "Yes"
   :doc:`hicpp-function-size <hicpp/function-size>`, :doc:`readability-function-size <readability/function-size>`,
   :doc:`hicpp-invalid-access-moved <hicpp/invalid-access-moved>`, :doc:`bugprone-use-after-move <bugprone/use-after-move>`,
   :doc:`hicpp-member-init <hicpp/member-init>`, :doc:`cppcoreguidelines-pro-type-member-init <cppcoreguidelines/pro-type-member-init>`, "Yes"
   :doc:`hicpp-move-const-arg <hicpp/move-const-arg>`, :doc:`performance-move-const-arg <performance/move-const-arg>`, "Yes"
   :doc:`hicpp-named-parameter <hicpp/named-parameter>`, :doc:`readability-named-parameter <readability/named-parameter>`, "Yes"
   :doc:`hicpp-new-delete-operators <hicpp/new-delete-operators>`, :doc:`misc-new-delete-overloads <misc/new-delete-overloads>`,
   :doc:`hicpp-no-array-decay <hicpp/no-array-decay>`, :doc:`cppcoreguidelines-pro-bounds-array-to-pointer-decay <cppcoreguidelines/pro-bounds-array-to-pointer-decay>`,
   :doc:`hicpp-no-malloc <hicpp/no-malloc>`, :doc:`cppcoreguidelines-no-malloc <cppcoreguidelines/no-malloc>`,
   :doc:`hicpp-noexcept-move <hicpp/noexcept-move>`, :doc:`performance-noexcept-move-constructor <performance/noexcept-move-constructor>`, "Yes"
   :doc:`hicpp-special-member-functions <hicpp/special-member-functions>`, :doc:`cppcoreguidelines-special-member-functions <cppcoreguidelines/special-member-functions>`,
   :doc:`hicpp-static-assert <hicpp/static-assert>`, :doc:`misc-static-assert <misc/static-assert>`, "Yes"
   :doc:`hicpp-undelegated-constructor <hicpp/undelegated-constructor>`, :doc:`bugprone-undelegated-constructor <bugprone/undelegated-constructor>`,
   :doc:`hicpp-uppercase-literal-suffix <hicpp/uppercase-literal-suffix>`, :doc:`readability-uppercase-literal-suffix <readability/uppercase-literal-suffix>`, "Yes"
   :doc:`hicpp-use-auto <hicpp/use-auto>`, :doc:`modernize-use-auto <modernize/use-auto>`, "Yes"
   :doc:`hicpp-use-emplace <hicpp/use-emplace>`, :doc:`modernize-use-emplace <modernize/use-emplace>`, "Yes"
   :doc:`hicpp-use-equals-default <hicpp/use-equals-default>`, :doc:`modernize-use-equals-default <modernize/use-equals-default>`, "Yes"
   :doc:`hicpp-use-equals-delete <hicpp/use-equals-delete>`, :doc:`modernize-use-equals-delete <modernize/use-equals-delete>`, "Yes"
   :doc:`hicpp-use-noexcept <hicpp/use-noexcept>`, :doc:`modernize-use-noexcept <modernize/use-noexcept>`, "Yes"
   :doc:`hicpp-use-nullptr <hicpp/use-nullptr>`, :doc:`modernize-use-nullptr <modernize/use-nullptr>`, "Yes"
   :doc:`hicpp-use-override <hicpp/use-override>`, :doc:`modernize-use-override <modernize/use-override>`, "Yes"
   :doc:`hicpp-vararg <hicpp/vararg>`, :doc:`cppcoreguidelines-pro-type-vararg <cppcoreguidelines/pro-type-vararg>`,
   :doc:`llvm-else-after-return <llvm/else-after-return>`, :doc:`readability-else-after-return <readability/else-after-return>`, "Yes"
   :doc:`llvm-qualified-auto <llvm/qualified-auto>`, :doc:`readability-qualified-auto <readability/qualified-auto>`, "Yes"
