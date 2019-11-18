.. title:: clang-tidy - Clang-Tidy Checks

Clang-Tidy Checks
=================

.. toctree::
   abseil-duration-addition
   abseil-duration-comparison
   abseil-duration-conversion-cast
   abseil-duration-division
   abseil-duration-factory-float
   abseil-duration-factory-scale
   abseil-duration-subtraction
   abseil-duration-unnecessary-conversion
   abseil-faster-strsplit-delimiter
   abseil-no-internal-dependencies
   abseil-no-namespace
   abseil-redundant-strcat-calls
   abseil-str-cat-append
   abseil-string-find-startswith
   abseil-time-comparison
   abseil-time-subtraction
   abseil-upgrade-duration-conversions
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
   android-cloexec-pipe
   android-cloexec-pipe2
   android-cloexec-socket
   android-comparison-in-temp-failure-retry
   boost-use-to-string
   bugprone-argument-comment
   bugprone-assert-side-effect
   bugprone-bad-signal-to-kill-thread
   bugprone-bool-pointer-implicit-conversion
   bugprone-branch-clone
   bugprone-copy-constructor-init
   bugprone-dangling-handle
   bugprone-dynamic-static-initializers
   bugprone-exception-escape
   bugprone-fold-init-type
   bugprone-forward-declaration-namespace
   bugprone-forwarding-reference-overload
   bugprone-inaccurate-erase
   bugprone-incorrect-roundings
   bugprone-infinite-loop
   bugprone-integer-division
   bugprone-lambda-function-name
   bugprone-macro-parentheses
   bugprone-macro-repeated-side-effects
   bugprone-misplaced-operator-in-strlen-in-alloc
   bugprone-misplaced-widening-cast
   bugprone-move-forwarding-reference
   bugprone-multiple-statement-macro
   bugprone-not-null-terminated-result
   bugprone-parent-virtual-call
   bugprone-posix-return
   bugprone-sizeof-container
   bugprone-sizeof-expression
   bugprone-string-constructor
   bugprone-string-integer-assignment
   bugprone-string-literal-with-embedded-nul
   bugprone-suspicious-enum-usage
   bugprone-suspicious-memset-usage
   bugprone-suspicious-missing-comma
   bugprone-suspicious-semicolon
   bugprone-suspicious-string-compare
   bugprone-swapped-arguments
   bugprone-terminating-continue
   bugprone-throw-keyword-missing
   bugprone-too-small-loop-variable
   bugprone-undefined-memory-manipulation
   bugprone-undelegated-constructor
   bugprone-unhandled-self-assignment
   bugprone-unused-raii
   bugprone-unused-return-value
   bugprone-use-after-move
   bugprone-virtual-near-miss
   cert-dcl03-c (redirects to misc-static-assert) <cert-dcl03-c>
   cert-dcl16-c (redirects to readability-uppercase-literal-suffix) <cert-dcl16-c>
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
   cert-mem57-cpp
   cert-msc30-c (redirects to cert-msc50-cpp) <cert-msc30-c>
   cert-msc32-c (redirects to cert-msc51-cpp) <cert-msc32-c>
   cert-msc50-cpp
   cert-msc51-cpp
   cert-oop11-cpp (redirects to performance-move-constructor-init) <cert-oop11-cpp>
   cert-oop54-cpp (redirects to bugprone-unhandled-self-assignment) <cert-oop54-cpp>
   cert-pos44-c (redirects to bugprone-bad-signal-to-kill-thread) <cert-pos44-c>
   clang-analyzer-core.CallAndMessage (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-core.CallAndMessage>
   clang-analyzer-core.DivideZero (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-core.DivideZero>
   clang-analyzer-core.DynamicTypePropagation
   clang-analyzer-core.NonNullParamChecker (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-core.NonNullParamChecker>
   clang-analyzer-core.NullDereference (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-core.NullDereference>
   clang-analyzer-core.StackAddressEscape (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-core.StackAddressEscape>
   clang-analyzer-core.UndefinedBinaryOperatorResult (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-core.UndefinedBinaryOperatorResult>
   clang-analyzer-core.VLASize (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-core.VLASize>
   clang-analyzer-core.uninitialized.ArraySubscript (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-core.uninitialized.ArraySubscript>
   clang-analyzer-core.uninitialized.Assign (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-core.uninitialized.Assign>
   clang-analyzer-core.uninitialized.Branch (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-core.uninitialized.Branch>
   clang-analyzer-core.uninitialized.CapturedBlockVariable
   clang-analyzer-core.uninitialized.UndefReturn (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-core.uninitialized.UndefReturn>
   clang-analyzer-cplusplus.InnerPointer
   clang-analyzer-cplusplus.Move (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-cplusplus.Move>
   clang-analyzer-cplusplus.NewDelete (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-cplusplus.NewDelete>
   clang-analyzer-cplusplus.NewDeleteLeaks (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-cplusplus.NewDeleteLeaks>
   clang-analyzer-deadcode.DeadStores (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-deadcode.DeadStores>
   clang-analyzer-nullability.NullPassedToNonnull (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-nullability.NullPassedToNonnull>
   clang-analyzer-nullability.NullReturnedFromNonnull (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-nullability.NullReturnedFromNonnull>
   clang-analyzer-nullability.NullableDereferenced (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-nullability.NullableDereferenced>
   clang-analyzer-nullability.NullablePassedToNonnull (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-nullability.NullablePassedToNonnull>
   clang-analyzer-nullability.NullableReturnedFromNonnull
   clang-analyzer-optin.cplusplus.UninitializedObject (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-optin.cplusplus.UninitializedObject>
   clang-analyzer-optin.cplusplus.VirtualCall (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-optin.cplusplus.VirtualCall>
   clang-analyzer-optin.mpi.MPI-Checker (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-optin.mpi.MPI-Checker>
   clang-analyzer-optin.osx.OSObjectCStyleCast
   clang-analyzer-optin.osx.cocoa.localizability.EmptyLocalizationContextChecker (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-optin.osx.cocoa.localizability.EmptyLocalizationContextChecker>
   clang-analyzer-optin.osx.cocoa.localizability.NonLocalizedStringChecker (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-optin.osx.cocoa.localizability.NonLocalizedStringChecker>
   clang-analyzer-optin.performance.GCDAntipattern
   clang-analyzer-optin.performance.Padding
   clang-analyzer-optin.portability.UnixAPI
   clang-analyzer-osx.API (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-osx.API>
   clang-analyzer-osx.MIG
   clang-analyzer-osx.NumberObjectConversion
   clang-analyzer-osx.OSObjectRetainCount
   clang-analyzer-osx.ObjCProperty
   clang-analyzer-osx.SecKeychainAPI (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-osx.SecKeychainAPI>
   clang-analyzer-osx.cocoa.AtSync (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-osx.cocoa.AtSync>
   clang-analyzer-osx.cocoa.AutoreleaseWrite
   clang-analyzer-osx.cocoa.ClassRelease (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-osx.cocoa.ClassRelease>
   clang-analyzer-osx.cocoa.Dealloc (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-osx.cocoa.Dealloc>
   clang-analyzer-osx.cocoa.IncompatibleMethodTypes (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-osx.cocoa.IncompatibleMethodTypes>
   clang-analyzer-osx.cocoa.Loops
   clang-analyzer-osx.cocoa.MissingSuperCall
   clang-analyzer-osx.cocoa.NSAutoreleasePool (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-osx.cocoa.NSAutoreleasePool>
   clang-analyzer-osx.cocoa.NSError (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-osx.cocoa.NSError>
   clang-analyzer-osx.cocoa.NilArg (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-osx.cocoa.NilArg>
   clang-analyzer-osx.cocoa.NonNilReturnValue
   clang-analyzer-osx.cocoa.ObjCGenerics (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-osx.cocoa.ObjCGenerics>
   clang-analyzer-osx.cocoa.RetainCount (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-osx.cocoa.RetainCount>
   clang-analyzer-osx.cocoa.RunLoopAutoreleaseLeak
   clang-analyzer-osx.cocoa.SelfInit (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-osx.cocoa.SelfInit>
   clang-analyzer-osx.cocoa.SuperDealloc (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-osx.cocoa.SuperDealloc>
   clang-analyzer-osx.cocoa.UnusedIvars (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-osx.cocoa.UnusedIvars>
   clang-analyzer-osx.cocoa.VariadicMethodTypes (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-osx.cocoa.VariadicMethodTypes>
   clang-analyzer-osx.coreFoundation.CFError (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-osx.coreFoundation.CFError>
   clang-analyzer-osx.coreFoundation.CFNumber (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-osx.coreFoundation.CFNumber>
   clang-analyzer-osx.coreFoundation.CFRetainRelease (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-osx.coreFoundation.CFRetainRelease>
   clang-analyzer-osx.coreFoundation.containers.OutOfBounds (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-osx.coreFoundation.containers.OutOfBounds>
   clang-analyzer-osx.coreFoundation.containers.PointerSizedValues (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-osx.coreFoundation.containers.PointerSizedValues>
   clang-analyzer-security.FloatLoopCounter (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-security.FloatLoopCounter>
   clang-analyzer-security.insecureAPI.DeprecatedOrUnsafeBufferHandling (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-security.insecureAPI.DeprecatedOrUnsafeBufferHandling>
   clang-analyzer-security.insecureAPI.UncheckedReturn (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-security.insecureAPI.UncheckedReturn>
   clang-analyzer-security.insecureAPI.bcmp (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-security.insecureAPI.bcmp>
   clang-analyzer-security.insecureAPI.bcopy (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-security.insecureAPI.bcopy>
   clang-analyzer-security.insecureAPI.bzero (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-security.insecureAPI.bzero>
   clang-analyzer-security.insecureAPI.getpw (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-security.insecureAPI.getpw>
   clang-analyzer-security.insecureAPI.gets (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-security.insecureAPI.gets>
   clang-analyzer-security.insecureAPI.mkstemp (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-security.insecureAPI.mkstemp>
   clang-analyzer-security.insecureAPI.mktemp (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-security.insecureAPI.mktemp>
   clang-analyzer-security.insecureAPI.rand (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-security.insecureAPI.rand>
   clang-analyzer-security.insecureAPI.strcpy (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-security.insecureAPI.strcpy>
   clang-analyzer-security.insecureAPI.vfork (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-security.insecureAPI.vfork>
   clang-analyzer-unix.API (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-unix.API>
   clang-analyzer-unix.Malloc (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-unix.Malloc>
   clang-analyzer-unix.MallocSizeof (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-unix.MallocSizeof>
   clang-analyzer-unix.MismatchedDeallocator (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-unix.MismatchedDeallocator>
   clang-analyzer-unix.Vfork (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-unix.Vfork>
   clang-analyzer-unix.cstring.BadSizeArg (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-unix.cstring.BadSizeArg>
   clang-analyzer-unix.cstring.NullArg (redirects to https://clang.llvm.org/docs/analyzer/checkers) <clang-analyzer-unix.cstring.NullArg>
   clang-analyzer-valist.CopyToSelf
   clang-analyzer-valist.Uninitialized
   clang-analyzer-valist.Unterminated
   cppcoreguidelines-avoid-c-arrays (redirects to modernize-avoid-c-arrays) <cppcoreguidelines-avoid-c-arrays>
   cppcoreguidelines-avoid-goto
   cppcoreguidelines-avoid-magic-numbers (redirects to readability-magic-numbers) <cppcoreguidelines-avoid-magic-numbers>
   cppcoreguidelines-c-copy-assignment-signature (redirects to misc-unconventional-assign-operator) <cppcoreguidelines-c-copy-assignment-signature>
   cppcoreguidelines-explicit-virtual-functions (redirects to modernize-use-override) <cppcoreguidelines-explicit-virtual-functions>
   cppcoreguidelines-init-variables
   cppcoreguidelines-interfaces-global-init
   cppcoreguidelines-macro-usage
   cppcoreguidelines-narrowing-conversions
   cppcoreguidelines-no-malloc
   cppcoreguidelines-non-private-member-variables-in-classes (redirects to misc-non-private-member-variables-in-classes) <cppcoreguidelines-non-private-member-variables-in-classes>
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
   darwin-avoid-spinlock
   darwin-dispatch-once-nonstatic
   fuchsia-default-arguments-calls
   fuchsia-default-arguments-declarations
   fuchsia-header-anon-namespaces (redirects to google-build-namespaces) <fuchsia-header-anon-namespaces>
   fuchsia-multiple-inheritance
   fuchsia-overloaded-operator
   fuchsia-restrict-system-includes
   fuchsia-statically-constructed-objects
   fuchsia-trailing-return
   fuchsia-virtual-inheritance
   google-build-explicit-make-pair
   google-build-namespaces
   google-build-using-namespace
   google-default-arguments
   google-explicit-constructor
   google-global-names-in-headers
   google-objc-avoid-nsobject-new
   google-objc-avoid-throwing-exception
   google-objc-function-naming
   google-objc-global-variable-declaration
   google-readability-avoid-underscore-in-googletest-name
   google-readability-braces-around-statements (redirects to readability-braces-around-statements) <google-readability-braces-around-statements>
   google-readability-casting
   google-readability-function-size (redirects to readability-function-size) <google-readability-function-size>
   google-readability-namespace-comments (redirects to llvm-namespace-comment) <google-readability-namespace-comments>
   google-readability-todo
   google-runtime-int
   google-runtime-operator
   google-runtime-references
   google-upgrade-googletest-case
   hicpp-avoid-c-arrays (redirects to modernize-avoid-c-arrays) <hicpp-avoid-c-arrays>
   hicpp-avoid-goto
   hicpp-braces-around-statements (redirects to readability-braces-around-statements) <hicpp-braces-around-statements>
   hicpp-deprecated-headers (redirects to modernize-deprecated-headers) <hicpp-deprecated-headers>
   hicpp-exception-baseclass
   hicpp-explicit-conversions (redirects to google-explicit-constructor) <hicpp-explicit-conversions>
   hicpp-function-size (redirects to readability-function-size) <hicpp-function-size>
   hicpp-invalid-access-moved (redirects to bugprone-use-after-move) <hicpp-invalid-access-moved>
   hicpp-member-init (redirects to cppcoreguidelines-pro-type-member-init) <hicpp-member-init>
   hicpp-move-const-arg (redirects to performance-move-const-arg) <hicpp-move-const-arg>
   hicpp-multiway-paths-covered
   hicpp-named-parameter (redirects to readability-named-parameter) <hicpp-named-parameter>
   hicpp-new-delete-operators (redirects to misc-new-delete-overloads) <hicpp-new-delete-operators>
   hicpp-no-array-decay (redirects to cppcoreguidelines-pro-bounds-array-to-pointer-decay) <hicpp-no-array-decay>
   hicpp-no-assembler
   hicpp-no-malloc (redirects to cppcoreguidelines-no-malloc) <hicpp-no-malloc>
   hicpp-noexcept-move (redirects to misc-noexcept-moveconstructor) <hicpp-noexcept-move>
   hicpp-signed-bitwise
   hicpp-special-member-functions (redirects to cppcoreguidelines-special-member-functions) <hicpp-special-member-functions>
   hicpp-static-assert (redirects to misc-static-assert) <hicpp-static-assert>
   hicpp-undelegated-constructor (redirects to bugprone-undelegated-constructor) <hicpp-undelegated-constructor>
   hicpp-uppercase-literal-suffix (redirects to readability-uppercase-literal-suffix) <hicpp-uppercase-literal-suffix>
   hicpp-use-auto (redirects to modernize-use-auto) <hicpp-use-auto>
   hicpp-use-emplace (redirects to modernize-use-emplace) <hicpp-use-emplace>
   hicpp-use-equals-default (redirects to modernize-use-equals-default) <hicpp-use-equals-default>
   hicpp-use-equals-delete (redirects to modernize-use-equals-delete) <hicpp-use-equals-delete>
   hicpp-use-noexcept (redirects to modernize-use-noexcept) <hicpp-use-noexcept>
   hicpp-use-nullptr (redirects to modernize-use-nullptr) <hicpp-use-nullptr>
   hicpp-use-override (redirects to modernize-use-override) <hicpp-use-override>
   hicpp-vararg (redirects to cppcoreguidelines-pro-type-vararg) <hicpp-vararg>
   linuxkernel-must-use-errs
   llvm-header-guard
   llvm-include-order
   llvm-namespace-comment
   llvm-prefer-isa-or-dyn-cast-in-conditionals
   llvm-prefer-register-over-unsigned
   llvm-twine-local
   misc-definitions-in-headers
   misc-misplaced-const
   misc-new-delete-overloads
   misc-non-copyable-objects
   misc-non-private-member-variables-in-classes
   misc-redundant-expression
   misc-static-assert
   misc-throw-by-value-catch-by-reference
   misc-unconventional-assign-operator
   misc-uniqueptr-reset-release
   misc-unused-alias-decls
   misc-unused-parameters
   misc-unused-using-decls
   modernize-avoid-bind
   modernize-avoid-c-arrays
   modernize-concat-nested-namespaces
   modernize-deprecated-headers
   modernize-deprecated-ios-base-aliases
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
   modernize-use-nodiscard
   modernize-use-noexcept
   modernize-use-nullptr
   modernize-use-override
   modernize-use-trailing-return-type
   modernize-use-transparent-functors
   modernize-use-uncaught-exceptions
   modernize-use-using
   mpi-buffer-deref
   mpi-type-mismatch
   objc-avoid-nserror-init
   objc-forbidden-subclassing
   objc-missing-hash
   objc-property-declaration
   objc-super-self
   openmp-exception-escape
   openmp-use-default-none
   performance-faster-string-find
   performance-for-range-copy
   performance-implicit-conversion-in-loop
   performance-inefficient-algorithm
   performance-inefficient-string-concatenation
   performance-inefficient-vector-operation
   performance-move-const-arg
   performance-move-constructor-init
   performance-no-automatic-move
   performance-noexcept-move-constructor
   performance-trivially-destructible
   performance-type-promotion-in-math-fn
   performance-unnecessary-copy-initialization
   performance-unnecessary-value-param
   portability-simd-intrinsics
   readability-avoid-const-params-in-decls
   readability-braces-around-statements
   readability-const-return-type
   readability-container-size-empty
   readability-convert-member-functions-to-static
   readability-delete-null-pointer
   readability-deleted-default
   readability-else-after-return
   readability-function-size
   readability-identifier-naming
   readability-implicit-bool-conversion
   readability-inconsistent-declaration-parameter-name
   readability-isolate-declaration
   readability-magic-numbers
   readability-make-member-function-const
   readability-misleading-indentation
   readability-misplaced-array-index
   readability-named-parameter
   readability-non-const-parameter
   readability-redundant-access-specifiers
   readability-redundant-control-flow
   readability-redundant-declaration
   readability-redundant-function-ptr-dereference
   readability-redundant-member-init
   readability-redundant-preprocessor
   readability-redundant-smartptr-get
   readability-redundant-string-cstr
   readability-redundant-string-init
   readability-simplify-boolean-expr
   readability-simplify-subscript-expr
   readability-static-accessed-through-instance
   readability-static-definition-in-anonymous-namespace
   readability-string-compare
   readability-uniqueptr-delete-release
   readability-uppercase-literal-suffix
   zircon-temporary-objects
