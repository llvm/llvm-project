.. title:: clang-tidy - Clang-Tidy Checks

Clang-Tidy Checks
=================

.. toctree::
   android-cloexec-accept
   android-cloexec-accept4
   android-cloexec-creat
   android-cloexec-dup
   android-cloexec-epoll-create
   android-cloexec-epoll-create1
   android-cloexec-fopen
   android-cloexec-inotify-init
   android-cloexec-inotify-init1
   android-cloexec-memfd-create
   android-cloexec-open
   android-cloexec-socket
   boost-use-to-string
   bugprone-argument-comment
   bugprone-assert-side-effect
   bugprone-bool-pointer-implicit-conversion
   bugprone-copy-constructor-init
   bugprone-dangling-handle
   bugprone-fold-init-type
   bugprone-forward-declaration-namespace
   bugprone-inaccurate-erase
   bugprone-incorrect-roundings
   bugprone-integer-division
   bugprone-misplaced-operator-in-strlen-in-alloc
   bugprone-move-forwarding-reference
   bugprone-multiple-statement-macro
   bugprone-string-constructor
   bugprone-suspicious-memset-usage
   bugprone-throw-keyword-missing
   bugprone-undefined-memory-manipulation
   bugprone-use-after-move
   bugprone-virtual-near-miss
   cert-dcl03-c (redirects to misc-static-assert) <cert-dcl03-c>
   cert-dcl21-cpp
   cert-dcl50-cpp
   cert-dcl54-cpp (redirects to misc-new-delete-overloads) <cert-dcl54-cpp>
   cert-dcl58-cpp
   cert-dcl59-cpp (redirects to google-build-namespaces) <cert-dcl59-cpp>
   cert-env33-c
   cert-err09-cpp (redirects to misc-throw-by-value-catch-by-reference) <cert-err09-cpp>
   cert-err34-c
   cert-err52-cpp
   cert-err58-cpp
   cert-err60-cpp
   cert-err61-cpp (redirects to misc-throw-by-value-catch-by-reference) <cert-err61-cpp>
   cert-fio38-c (redirects to misc-non-copyable-objects) <cert-fio38-c>
   cert-flp30-c
   cert-msc30-c (redirects to cert-msc50-cpp) <cert-msc30-c>
   cert-msc50-cpp
   cert-oop11-cpp (redirects to performance-move-constructor-init) <cert-oop11-cpp>
   cppcoreguidelines-avoid-goto
   cppcoreguidelines-c-copy-assignment-signature (redirects to misc-unconventional-assign-operator) <cppcoreguidelines-c-copy-assignment-signature>
   cppcoreguidelines-interfaces-global-init
   cppcoreguidelines-no-malloc
   cppcoreguidelines-owning-memory
   cppcoreguidelines-pro-bounds-array-to-pointer-decay
   cppcoreguidelines-pro-bounds-constant-array-index
   cppcoreguidelines-pro-bounds-pointer-arithmetic
   cppcoreguidelines-pro-type-const-cast
   cppcoreguidelines-pro-type-cstyle-cast
   cppcoreguidelines-pro-type-member-init
   cppcoreguidelines-pro-type-reinterpret-cast
   cppcoreguidelines-pro-type-static-cast-downcast
   cppcoreguidelines-pro-type-union-access
   cppcoreguidelines-pro-type-vararg
   cppcoreguidelines-slicing
   cppcoreguidelines-special-member-functions
   fuchsia-default-arguments
   fuchsia-multiple-inheritance
   fuchsia-overloaded-operator
   fuchsia-statically-constructed-objects
   fuchsia-trailing-return
   fuchsia-virtual-inheritance
   google-build-explicit-make-pair
   google-build-namespaces
   google-build-using-namespace
   google-default-arguments
   google-explicit-constructor
   google-global-names-in-headers
   google-objc-avoid-throwing-exception
   google-objc-global-variable-declaration
   google-readability-braces-around-statements (redirects to readability-braces-around-statements) <google-readability-braces-around-statements>
   google-readability-casting
   google-readability-function-size (redirects to readability-function-size) <google-readability-function-size>
   google-readability-namespace-comments (redirects to llvm-namespace-comment) <google-readability-namespace-comments>
   google-readability-redundant-smartptr-get (redirects to readability-redundant-smartptr-get) <google-readability-redundant-smartptr-get>
   google-readability-todo
   google-runtime-int
   google-runtime-member-string-references
   google-runtime-operator
   google-runtime-references
   hicpp-avoid-goto
   hicpp-braces-around-statements (redirects to readability-braces-around-statements) <hicpp-braces-around-statements>
   hicpp-deprecated-headers (redirects to modernize-deprecated-headers) <hicpp-deprecated-headers>
   hicpp-exception-baseclass
   hicpp-explicit-conversions (redirects to google-explicit-constructor) <hicpp-explicit-conversions>
   hicpp-function-size (redirects to readability-function-size) <hicpp-function-size>
   hicpp-invalid-access-moved (redirects to bugprone-use-after-move) <hicpp-invalid-access-moved>
   hicpp-member-init (redirects to cppcoreguidelines-pro-type-member-init) <hicpp-member-init>
   hicpp-move-const-arg (redirects to performance-move-const-arg) <hicpp-move-const-arg>
   hicpp-named-parameter (redirects to readability-named-parameter) <hicpp-named-parameter>
   hicpp-new-delete-operators (redirects to misc-new-delete-overloads) <hicpp-new-delete-operators>
   hicpp-no-array-decay (redirects to cppcoreguidelines-pro-bounds-array-to-pointer-decay) <hicpp-no-array-decay>
   hicpp-no-assembler
   hicpp-no-malloc (redirects to cppcoreguidelines-no-malloc) <hicpp-no-malloc>
   hicpp-noexcept-move (redirects to misc-noexcept-moveconstructor) <hicpp-noexcept-move>
   hicpp-signed-bitwise
   hicpp-special-member-functions (redirects to cppcoreguidelines-special-member-functions) <hicpp-special-member-functions>
   hicpp-static-assert (redirects to misc-static-assert) <hicpp-static-assert>
   hicpp-undelegated-constructor (redirects to misc-undelegated-constructor) <hicpp-undelegated-constructor>
   hicpp-use-auto (redirects to modernize-use-auto) <hicpp-use-auto>
   hicpp-use-emplace (redirects to modernize-use-emplace) <hicpp-use-emplace>
   hicpp-use-equals-default (redirects to modernize-use-equals-default) <hicpp-use-equals-default>
   hicpp-use-equals-delete (redirects to modernize-use-equals-delete) <hicpp-use-equals-delete>
   hicpp-use-noexcept (redirects to modernize-use-noexcept) <hicpp-use-noexcept>
   hicpp-use-nullptr (redirects to modernize-use-nullptr) <hicpp-use-nullptr>
   hicpp-use-override (redirects to modernize-use-override) <hicpp-use-override>
   hicpp-vararg (redirects to cppcoreguidelines-pro-type-vararg) <hicpp-vararg>
   llvm-header-guard
   llvm-include-order
   llvm-namespace-comment
   llvm-twine-local
   misc-definitions-in-headers
   misc-forwarding-reference-overload
   misc-lambda-function-name
   misc-macro-parentheses
   misc-macro-repeated-side-effects
   misc-misplaced-const
   misc-misplaced-widening-cast
   misc-new-delete-overloads
   misc-non-copyable-objects
   misc-redundant-expression
   misc-sizeof-container
   misc-sizeof-expression
   misc-static-assert
   misc-string-integer-assignment
   misc-string-literal-with-embedded-nul
   misc-suspicious-enum-usage
   misc-suspicious-missing-comma
   misc-suspicious-semicolon
   misc-suspicious-string-compare
   misc-swapped-arguments
   misc-throw-by-value-catch-by-reference
   misc-unconventional-assign-operator
   misc-undelegated-constructor
   misc-uniqueptr-reset-release
   misc-unused-alias-decls
   misc-unused-parameters
   misc-unused-raii
   misc-unused-using-decls
   modernize-avoid-bind
   modernize-deprecated-headers
   modernize-loop-convert
   modernize-make-shared
   modernize-make-unique
   modernize-pass-by-value
   modernize-raw-string-literal
   modernize-redundant-void-arg
   modernize-replace-auto-ptr
   modernize-replace-random-shuffle
   modernize-return-braced-init-list
   modernize-shrink-to-fit
   modernize-unary-static-assert
   modernize-use-auto
   modernize-use-bool-literals
   modernize-use-default-member-init
   modernize-use-emplace
   modernize-use-equals-default
   modernize-use-equals-delete
   modernize-use-noexcept
   modernize-use-nullptr
   modernize-use-override
   modernize-use-transparent-functors
   modernize-use-uncaught-exceptions
   modernize-use-using
   mpi-buffer-deref
   mpi-type-mismatch
   objc-avoid-nserror-init
   objc-avoid-spinlock
   objc-forbidden-subclassing
   objc-property-declaration
   performance-faster-string-find
   performance-for-range-copy
   performance-implicit-conversion-in-loop
   performance-inefficient-algorithm
   performance-inefficient-string-concatenation
   performance-inefficient-vector-operation
   performance-move-const-arg
   performance-move-constructor-init
   performance-noexcept-move-constructor
   performance-type-promotion-in-math-fn
   performance-unnecessary-copy-initialization
   performance-unnecessary-value-param
   readability-avoid-const-params-in-decls
   readability-braces-around-statements
   readability-container-size-empty
   readability-delete-null-pointer
   readability-deleted-default
   readability-else-after-return
   readability-function-size
   readability-identifier-naming
   readability-implicit-bool-conversion
   readability-inconsistent-declaration-parameter-name
   readability-misleading-indentation
   readability-misplaced-array-index
   readability-named-parameter
   readability-non-const-parameter
   readability-redundant-control-flow
   readability-redundant-declaration
   readability-redundant-function-ptr-dereference
   readability-redundant-member-init
   readability-redundant-smartptr-get
   readability-redundant-string-cstr
   readability-redundant-string-init
   readability-simd-intrinsics
   readability-simplify-boolean-expr
   readability-static-accessed-through-instance
   readability-static-definition-in-anonymous-namespace
   readability-string-compare
   readability-uniqueptr-delete-release
