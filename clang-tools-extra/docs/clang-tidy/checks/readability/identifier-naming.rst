.. title:: clang-tidy - readability-identifier-naming

readability-identifier-naming
=============================

Checks for identifiers naming style mismatch.

This check will try to enforce coding guidelines on the identifiers naming. It
supports one of the following casing types and tries to convert from one to
another if a mismatch is detected

Casing types include:

 - ``lower_case``
 - ``UPPER_CASE``
 - ``camelBack``
 - ``CamelCase``
 - ``camel_Snake_Back``
 - ``Camel_Snake_Case``
 - ``aNy_CasE``
 - ``Leading_upper_snake_case``

It also supports a fixed prefix and suffix that will be prepended or appended
to the identifiers, regardless of the casing.

Many configuration options are available, in order to be able to create
different rules for different kinds of identifiers. In general, the rules are
falling back to a more generic rule if the specific case is not configured.

The naming of virtual methods is reported where they occur in the base class,
but not where they are overridden, as it can't be fixed locally there.
This also applies for pseudo-override patterns like CRTP.

``Leading_upper_snake_case`` is a naming convention where the first word is capitalized
followed by lower case word(s) separated by underscore(s) '_'. Examples include:
`Cap_snake_case`, `Cobra_case`, `Foo_bar_baz`, and `Master_copy_8gb`.

Hungarian notation can be customized using different *HungarianPrefix* settings.
The options and their corresponding values are:

 - ``Off`` - the default setting
 - ``On`` - example: ``int iVariable``
 - ``LowerCase`` - example: ``int i_Variable``
 - ``CamelCase`` - example: ``int IVariable``

Options summary
---------------

The available options are summarized below:

**General options**

 - :option:`AggressiveDependentMemberLookup`
 - :option:`CheckAnonFieldInParent`
 - :option:`GetConfigPerFile`
 - :option:`IgnoreMainLikeFunctions`

**Specific options**

 - :option:`AbstractClassCase`, :option:`AbstractClassPrefix`, :option:`AbstractClassSuffix`, :option:`AbstractClassIgnoredRegexp`, :option:`AbstractClassHungarianPrefix`
 - :option:`ClassCase`, :option:`ClassPrefix`, :option:`ClassSuffix`, :option:`ClassIgnoredRegexp`, :option:`ClassHungarianPrefix`
 - :option:`ClassConstantCase`, :option:`ClassConstantPrefix`, :option:`ClassConstantSuffix`, :option:`ClassConstantIgnoredRegexp`, :option:`ClassConstantHungarianPrefix`
 - :option:`ClassMemberCase`, :option:`ClassMemberPrefix`, :option:`ClassMemberSuffix`, :option:`ClassMemberIgnoredRegexp`, :option:`ClassMemberHungarianPrefix`
 - :option:`ClassMethodCase`, :option:`ClassMethodPrefix`, :option:`ClassMethodSuffix`, :option:`ClassMethodIgnoredRegexp`
 - :option:`ConceptCase`, :option:`ConceptPrefix`, :option:`ConceptSuffix`, :option:`ConceptIgnoredRegexp`
 - :option:`ConstantCase`, :option:`ConstantPrefix`, :option:`ConstantSuffix`, :option:`ConstantIgnoredRegexp`, :option:`ConstantHungarianPrefix`
 - :option:`ConstantMemberCase`, :option:`ConstantMemberPrefix`, :option:`ConstantMemberSuffix`, :option:`ConstantMemberIgnoredRegexp`, :option:`ConstantMemberHungarianPrefix`
 - :option:`ConstantParameterCase`, :option:`ConstantParameterPrefix`, :option:`ConstantParameterSuffix`, :option:`ConstantParameterIgnoredRegexp`, :option:`ConstantParameterHungarianPrefix`
 - :option:`ConstantPointerParameterCase`, :option:`ConstantPointerParameterPrefix`, :option:`ConstantPointerParameterSuffix`, :option:`ConstantPointerParameterIgnoredRegexp`, :option:`ConstantPointerParameterHungarianPrefix`
 - :option:`ConstexprFunctionCase`, :option:`ConstexprFunctionPrefix`, :option:`ConstexprFunctionSuffix`, :option:`ConstexprFunctionIgnoredRegexp`
 - :option:`ConstexprMethodCase`, :option:`ConstexprMethodPrefix`, :option:`ConstexprMethodSuffix`, :option:`ConstexprMethodIgnoredRegexp`
 - :option:`ConstexprVariableCase`, :option:`ConstexprVariablePrefix`, :option:`ConstexprVariableSuffix`, :option:`ConstexprVariableIgnoredRegexp`, :option:`ConstexprVariableHungarianPrefix`
 - :option:`EnumCase`, :option:`EnumPrefix`, :option:`EnumSuffix`, :option:`EnumIgnoredRegexp`
 - :option:`EnumConstantCase`, :option:`EnumConstantPrefix`, :option:`EnumConstantSuffix`, :option:`EnumConstantIgnoredRegexp`, :option:`EnumConstantHungarianPrefix`
 - :option:`FunctionCase`, :option:`FunctionPrefix`, :option:`FunctionSuffix`, :option:`FunctionIgnoredRegexp`
 - :option:`GlobalConstantCase`, :option:`GlobalConstantPrefix`, :option:`GlobalConstantSuffix`, :option:`GlobalConstantIgnoredRegexp`, :option:`GlobalConstantHungarianPrefix`
 - :option:`GlobalConstantPointerCase`, :option:`GlobalConstantPointerPrefix`, :option:`GlobalConstantPointerSuffix`, :option:`GlobalConstantPointerIgnoredRegexp`, :option:`GlobalConstantPointerHungarianPrefix`
 - :option:`GlobalFunctionCase`, :option:`GlobalFunctionPrefix`, :option:`GlobalFunctionSuffix`, :option:`GlobalFunctionIgnoredRegexp`
 - :option:`GlobalPointerCase`, :option:`GlobalPointerPrefix`, :option:`GlobalPointerSuffix`, :option:`GlobalPointerIgnoredRegexp`, :option:`GlobalPointerHungarianPrefix`
 - :option:`GlobalVariableCase`, :option:`GlobalVariablePrefix`, :option:`GlobalVariableSuffix`, :option:`GlobalVariableIgnoredRegexp`, :option:`GlobalVariableHungarianPrefix`
 - :option:`InlineNamespaceCase`, :option:`InlineNamespacePrefix`, :option:`InlineNamespaceSuffix`, :option:`InlineNamespaceIgnoredRegexp`
 - :option:`LocalConstantCase`, :option:`LocalConstantPrefix`, :option:`LocalConstantSuffix`, :option:`LocalConstantIgnoredRegexp`, :option:`LocalConstantHungarianPrefix`
 - :option:`LocalConstantPointerCase`, :option:`LocalConstantPointerPrefix`, :option:`LocalConstantPointerSuffix`, :option:`LocalConstantPointerIgnoredRegexp`, :option:`LocalConstantPointerHungarianPrefix`
 - :option:`LocalPointerCase`, :option:`LocalPointerPrefix`, :option:`LocalPointerSuffix`, :option:`LocalPointerIgnoredRegexp`, :option:`LocalPointerHungarianPrefix`
 - :option:`LocalVariableCase`, :option:`LocalVariablePrefix`, :option:`LocalVariableSuffix`, :option:`LocalVariableIgnoredRegexp`, :option:`LocalVariableHungarianPrefix`
 - :option:`MacroDefinitionCase`, :option:`MacroDefinitionPrefix`, :option:`MacroDefinitionSuffix`, :option:`MacroDefinitionIgnoredRegexp`
 - :option:`MemberCase`, :option:`MemberPrefix`, :option:`MemberSuffix`, :option:`MemberIgnoredRegexp`, :option:`MemberHungarianPrefix`
 - :option:`MethodCase`, :option:`MethodPrefix`, :option:`MethodSuffix`, :option:`MethodIgnoredRegexp`
 - :option:`NamespaceCase`, :option:`NamespacePrefix`, :option:`NamespaceSuffix`, :option:`NamespaceIgnoredRegexp`
 - :option:`ParameterCase`, :option:`ParameterPrefix`, :option:`ParameterSuffix`, :option:`ParameterIgnoredRegexp`, :option:`ParameterHungarianPrefix`
 - :option:`ParameterPackCase`, :option:`ParameterPackPrefix`, :option:`ParameterPackSuffix`, :option:`ParameterPackIgnoredRegexp`
 - :option:`PointerParameterCase`, :option:`PointerParameterPrefix`, :option:`PointerParameterSuffix`, :option:`PointerParameterIgnoredRegexp`, :option:`PointerParameterHungarianPrefix`
 - :option:`PrivateMemberCase`, :option:`PrivateMemberPrefix`, :option:`PrivateMemberSuffix`, :option:`PrivateMemberIgnoredRegexp`, :option:`PrivateMemberHungarianPrefix`
 - :option:`PrivateMethodCase`, :option:`PrivateMethodPrefix`, :option:`PrivateMethodSuffix`, :option:`PrivateMethodIgnoredRegexp`
 - :option:`ProtectedMemberCase`, :option:`ProtectedMemberPrefix`, :option:`ProtectedMemberSuffix`, :option:`ProtectedMemberIgnoredRegexp`, :option:`ProtectedMemberHungarianPrefix`
 - :option:`ProtectedMethodCase`, :option:`ProtectedMethodPrefix`, :option:`ProtectedMethodSuffix`, :option:`ProtectedMethodIgnoredRegexp`
 - :option:`PublicMemberCase`, :option:`PublicMemberPrefix`, :option:`PublicMemberSuffix`, :option:`PublicMemberIgnoredRegexp`, :option:`PublicMemberHungarianPrefix`
 - :option:`PublicMethodCase`, :option:`PublicMethodPrefix`, :option:`PublicMethodSuffix`, :option:`PublicMethodIgnoredRegexp`
 - :option:`ScopedEnumConstantCase`, :option:`ScopedEnumConstantPrefix`, :option:`ScopedEnumConstantSuffix`, :option:`ScopedEnumConstantIgnoredRegexp`
 - :option:`StaticConstantCase`, :option:`StaticConstantPrefix`, :option:`StaticConstantSuffix`, :option:`StaticConstantIgnoredRegexp`, :option:`StaticConstantHungarianPrefix`
 - :option:`StaticVariableCase`, :option:`StaticVariablePrefix`, :option:`StaticVariableSuffix`, :option:`StaticVariableIgnoredRegexp`, :option:`StaticVariableHungarianPrefix`
 - :option:`StructCase`, :option:`StructPrefix`, :option:`StructSuffix`, :option:`StructIgnoredRegexp`
 - :option:`TemplateParameterCase`, :option:`TemplateParameterPrefix`, :option:`TemplateParameterSuffix`, :option:`TemplateParameterIgnoredRegexp`
 - :option:`TemplateTemplateParameterCase`, :option:`TemplateTemplateParameterPrefix`, :option:`TemplateTemplateParameterSuffix`, :option:`TemplateTemplateParameterIgnoredRegexp`
 - :option:`TypeAliasCase`, :option:`TypeAliasPrefix`, :option:`TypeAliasSuffix`, :option:`TypeAliasIgnoredRegexp`
 - :option:`TypedefCase`, :option:`TypedefPrefix`, :option:`TypedefSuffix`, :option:`TypedefIgnoredRegexp`
 - :option:`TypeTemplateParameterCase`, :option:`TypeTemplateParameterPrefix`, :option:`TypeTemplateParameterSuffix`, :option:`TypeTemplateParameterIgnoredRegexp`
 - :option:`UnionCase`, :option:`UnionPrefix`, :option:`UnionSuffix`, :option:`UnionIgnoredRegexp`
 - :option:`ValueTemplateParameterCase`, :option:`ValueTemplateParameterPrefix`, :option:`ValueTemplateParameterSuffix`, :option:`ValueTemplateParameterIgnoredRegexp`
 - :option:`VariableCase`, :option:`VariablePrefix`, :option:`VariableSuffix`, :option:`VariableIgnoredRegexp`, :option:`VariableHungarianPrefix`
 - :option:`VirtualMethodCase`, :option:`VirtualMethodPrefix`, :option:`VirtualMethodSuffix`, :option:`VirtualMethodIgnoredRegexp`


Options description
-------------------

A detailed description of each option is presented below:

.. option:: AbstractClassCase (added in 15.0.0)

    When defined, the check will ensure abstract class names conform to the
    selected casing.

.. option:: AbstractClassPrefix (added in 15.0.0)

    When defined, the check will ensure abstract class names will add the
    prefixed with the given value (regardless of casing).

.. option:: AbstractClassIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for abstract class names
    matching this regular expression.

.. option:: AbstractClassSuffix (added in 15.0.0)

    When defined, the check will ensure abstract class names will add the
    suffix with the given value (regardless of casing).

.. option:: AbstractClassHungarianPrefix (added in 15.0.0)

    When enabled, the check ensures that the declared identifier will
    have a Hungarian notation prefix based on the declared type.

For example using values of:

   - AbstractClassCase of ``lower_case``
   - AbstractClassPrefix of ``pre_``
   - AbstractClassSuffix of ``_post``
   - AbstractClassHungarianPrefix of ``On``


Identifies and/or transforms abstract class names as follows:

Before:

.. code-block:: c++

    class ABSTRACT_CLASS {
    public:
      ABSTRACT_CLASS();
    };

After:

.. code-block:: c++

    class pre_abstract_class_post {
    public:
      pre_abstract_class_post();
    };

.. option:: AggressiveDependentMemberLookup (added in 15.0.0)

    When set to `true` the check will look in dependent base classes for dependent
    member references that need changing. This can lead to errors with template
    specializations so the default value is `false`.

For example using values of:

   - ClassMemberCase of ``lower_case``

Before:

.. code-block:: c++

    template <typename T>
    struct Base {
      T BadNamedMember;
    };

    template <typename T>
    struct Derived : Base<T> {
      void reset() {
        this->BadNamedMember = 0;
      }
    };

After if AggressiveDependentMemberLookup is `false`:

.. code-block:: c++

    template <typename T>
    struct Base {
      T bad_named_member;
    };

    template <typename T>
    struct Derived : Base<T> {
      void reset() {
        this->BadNamedMember = 0;
      }
    };

After if AggressiveDependentMemberLookup is `true`:

.. code-block:: c++

    template <typename T>
    struct Base {
      T bad_named_member;
    };

    template <typename T>
    struct Derived : Base<T> {
      void reset() {
        this->bad_named_member = 0;
      }
    };

.. option:: CheckAnonFieldInParent (added in 18.1.0)

    When set to `true`, fields in anonymous records (i.e. anonymous
    unions and structs) will be treated as names in the enclosing scope
    rather than public members of the anonymous record for the purpose
    of name checking.

For example:

.. code-block:: c++

    class Foo {
    private:
      union {
        int iv_;
        float fv_;
      };
    };

If :option:`CheckAnonFieldInParent` is `false`, you may get warnings
that ``iv_`` and ``fv_`` are not coherent to public member names, because
``iv_`` and ``fv_`` are public members of the anonymous union. When
:option:`CheckAnonFieldInParent` is `true`, ``iv_`` and ``fv_`` will be
treated as private data members of ``Foo`` for the purpose of name checking
and thus no warnings will be emitted.

.. option:: ClassCase (added in 15.0.0)

    When defined, the check will ensure class names conform to the
    selected casing.

.. option:: ClassPrefix (added in 15.0.0)

    When defined, the check will ensure class names will add the
    prefixed with the given value (regardless of casing).

.. option:: ClassIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for class names matching
    this regular expression.

.. option:: ClassSuffix (added in 15.0.0)

    When defined, the check will ensure class names will add the
    suffix with the given value (regardless of casing).

.. option:: ClassHungarianPrefix (added in 15.0.0)

    When enabled, the check ensures that the declared identifier will
    have a Hungarian notation prefix based on the declared type.

For example using values of:

   - ClassCase of ``lower_case``
   - ClassPrefix of ``pre_``
   - ClassSuffix of ``_post``
   - ClassHungarianPrefix of ``On``

Identifies and/or transforms class names as follows:

Before:

.. code-block:: c++

    class FOO {
    public:
      FOO();
      ~FOO();
    };

After:

.. code-block:: c++

    class pre_foo_post {
    public:
      pre_foo_post();
      ~pre_foo_post();
    };

.. option:: ClassConstantCase (added in 15.0.0)

    When defined, the check will ensure class constant names conform to the
    selected casing.

.. option:: ClassConstantPrefix (added in 15.0.0)

    When defined, the check will ensure class constant names will add the
    prefixed with the given value (regardless of casing).

.. option:: ClassConstantIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for class constant names
    matching this regular expression.

.. option:: ClassConstantSuffix (added in 15.0.0)

    When defined, the check will ensure class constant names will add the
    suffix with the given value (regardless of casing).

.. option:: ClassConstantHungarianPrefix (added in 15.0.0)

    When enabled, the check ensures that the declared identifier will
    have a Hungarian notation prefix based on the declared type.

For example using values of:

   - ClassConstantCase of ``lower_case``
   - ClassConstantPrefix of ``pre_``
   - ClassConstantSuffix of ``_post``
   - ClassConstantHungarianPrefix of ``On``

Identifies and/or transforms class constant names as follows:

Before:

.. code-block:: c++

    class FOO {
    public:
      static const int CLASS_CONSTANT;
    };

After:

.. code-block:: c++

    class FOO {
    public:
      static const int pre_class_constant_post;
    };

.. option:: ClassMemberCase (added in 15.0.0)

    When defined, the check will ensure class member names conform to the
    selected casing.

.. option:: ClassMemberPrefix (added in 15.0.0)

    When defined, the check will ensure class member names will add the
    prefixed with the given value (regardless of casing).

.. option:: ClassMemberIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for class member names
    matching this regular expression.

.. option:: ClassMemberSuffix (added in 15.0.0)

    When defined, the check will ensure class member names will add the
    suffix with the given value (regardless of casing).

.. option:: ClassMemberHungarianPrefix (added in 15.0.0)

    When enabled, the check ensures that the declared identifier will
    have a Hungarian notation prefix based on the declared type.

For example using values of:

   - ClassMemberCase of ``lower_case``
   - ClassMemberPrefix of ``pre_``
   - ClassMemberSuffix of ``_post``
   - ClassMemberHungarianPrefix of ``On``

Identifies and/or transforms class member names as follows:

Before:

.. code-block:: c++

    class FOO {
    public:
      static int CLASS_CONSTANT;
    };

After:

.. code-block:: c++

    class FOO {
    public:
      static int pre_class_constant_post;
    };

.. option:: ClassMethodCase (added in 15.0.0)

    When defined, the check will ensure class method names conform to the
    selected casing.

.. option:: ClassMethodPrefix (added in 15.0.0)

    When defined, the check will ensure class method names will add the
    prefixed with the given value (regardless of casing).

.. option:: ClassMethodIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for class method names
    matching this regular expression.

.. option:: ClassMethodSuffix (added in 15.0.0)

    When defined, the check will ensure class method names will add the
    suffix with the given value (regardless of casing).

For example using values of:

   - ClassMethodCase of ``lower_case``
   - ClassMethodPrefix of ``pre_``
   - ClassMethodSuffix of ``_post``

Identifies and/or transforms class method names as follows:

Before:

.. code-block:: c++

    class FOO {
    public:
      int CLASS_MEMBER();
    };

After:

.. code-block:: c++

    class FOO {
    public:
      int pre_class_member_post();
    };

.. option:: ConceptCase (added in 18.1.0)

    When defined, the check will ensure concept names conform to the
    selected casing.

.. option:: ConceptPrefix (added in 18.1.0)

    When defined, the check will ensure concept names will add the
    prefixed with the given value (regardless of casing).

.. option:: ConceptIgnoredRegexp (added in 18.1.0)

    Identifier naming checks won't be enforced for concept names
    matching this regular expression.

.. option:: ConceptSuffix (added in 18.1.0)

    When defined, the check will ensure concept names will add the
    suffix with the given value (regardless of casing).

For example using values of:

   - ConceptCase of ``CamelCase``
   - ConceptPrefix of ``Pre``
   - ConceptSuffix of ``Post``

Identifies and/or transforms concept names as follows:

Before:

.. code-block:: c++

    template<typename T> concept my_concept = requires (T t) { {t++}; };

After:

.. code-block:: c++

    template<typename T> concept PreMyConceptPost = requires (T t) { {t++}; };

.. option:: ConstantCase (added in 15.0.0)

    When defined, the check will ensure constant names conform to the
    selected casing.

.. option:: ConstantPrefix (added in 15.0.0)

    When defined, the check will ensure constant names will add the
    prefixed with the given value (regardless of casing).

.. option:: ConstantIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for constant names
    matching this regular expression.

.. option:: ConstantSuffix (added in 15.0.0)

    When defined, the check will ensure constant names will add the
    suffix with the given value (regardless of casing).

.. option:: ConstantHungarianPrefix (added in 15.0.0)

    When enabled, the check ensures that the declared identifier will
    have a Hungarian notation prefix based on the declared type.

For example using values of:

   - ConstantCase of ``lower_case``
   - ConstantPrefix of ``pre_``
   - ConstantSuffix of ``_post``
   - ConstantHungarianPrefix of ``On``

Identifies and/or transforms constant names as follows:

Before:

.. code-block:: c++

    void function() { unsigned const MyConst_array[] = {1, 2, 3}; }

After:

.. code-block:: c++

    void function() { unsigned const pre_myconst_array_post[] = {1, 2, 3}; }

.. option:: ConstantMemberCase (added in 15.0.0)

    When defined, the check will ensure constant member names conform to the
    selected casing.

.. option:: ConstantMemberPrefix (added in 15.0.0)

    When defined, the check will ensure constant member names will add the
    prefixed with the given value (regardless of casing).

.. option:: ConstantMemberIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for constant member names
    matching this regular expression.

.. option:: ConstantMemberSuffix (added in 15.0.0)

    When defined, the check will ensure constant member names will add the
    suffix with the given value (regardless of casing).

.. option:: ConstantMemberHungarianPrefix (added in 15.0.0)

    When enabled, the check ensures that the declared identifier will
    have a Hungarian notation prefix based on the declared type.

For example using values of:

   - ConstantMemberCase of ``lower_case``
   - ConstantMemberPrefix of ``pre_``
   - ConstantMemberSuffix of ``_post``
   - ConstantMemberHungarianPrefix of ``On``

Identifies and/or transforms constant member names as follows:

Before:

.. code-block:: c++

    class Foo {
      char const MY_ConstMember_string[4] = "123";
    }

After:

.. code-block:: c++

    class Foo {
      char const pre_my_constmember_string_post[4] = "123";
    }

.. option:: ConstantParameterCase (added in 15.0.0)

    When defined, the check will ensure constant parameter names conform to the
    selected casing.

.. option:: ConstantParameterPrefix (added in 15.0.0)

    When defined, the check will ensure constant parameter names will add the
    prefixed with the given value (regardless of casing).

.. option:: ConstantParameterIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for constant parameter names
    matching this regular expression.

.. option:: ConstantParameterSuffix (added in 15.0.0)

    When defined, the check will ensure constant parameter names will add the
    suffix with the given value (regardless of casing).

.. option:: ConstantParameterHungarianPrefix (added in 15.0.0)

    When enabled, the check ensures that the declared identifier will
    have a Hungarian notation prefix based on the declared type.

For example using values of:

   - ConstantParameterCase of ``lower_case``
   - ConstantParameterPrefix of ``pre_``
   - ConstantParameterSuffix of ``_post``
   - ConstantParameterHungarianPrefix of ``On``

Identifies and/or transforms constant parameter names as follows:

Before:

.. code-block:: c++

    void GLOBAL_FUNCTION(int PARAMETER_1, int const CONST_parameter);

After:

.. code-block:: c++

    void GLOBAL_FUNCTION(int PARAMETER_1, int const pre_const_parameter_post);

.. option:: ConstantPointerParameterCase (added in 15.0.0)

    When defined, the check will ensure constant pointer parameter names conform to the
    selected casing.

.. option:: ConstantPointerParameterPrefix (added in 15.0.0)

    When defined, the check will ensure constant pointer parameter names will add the
    prefixed with the given value (regardless of casing).

.. option:: ConstantPointerParameterIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for constant pointer parameter
    names matching this regular expression.

.. option:: ConstantPointerParameterSuffix (added in 15.0.0)

    When defined, the check will ensure constant pointer parameter names will add the
    suffix with the given value (regardless of casing).

.. option:: ConstantPointerParameterHungarianPrefix (added in 15.0.0)

    When enabled, the check ensures that the declared identifier will
    have a Hungarian notation prefix based on the declared type.

For example using values of:

   - ConstantPointerParameterCase of ``lower_case``
   - ConstantPointerParameterPrefix of ``pre_``
   - ConstantPointerParameterSuffix of ``_post``
   - ConstantPointerParameterHungarianPrefix of ``On``

Identifies and/or transforms constant pointer parameter names as follows:

Before:

.. code-block:: c++

    void GLOBAL_FUNCTION(int const *CONST_parameter);

After:

.. code-block:: c++

    void GLOBAL_FUNCTION(int const *pre_const_parameter_post);

.. option:: ConstexprFunctionCase (added in 15.0.0)

    When defined, the check will ensure constexpr function names conform to the
    selected casing.

.. option:: ConstexprFunctionPrefix (added in 15.0.0)

    When defined, the check will ensure constexpr function names will add the
    prefixed with the given value (regardless of casing).

.. option:: ConstexprFunctionIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for constexpr function names
    matching this regular expression.

.. option:: ConstexprFunctionSuffix (added in 15.0.0)

    When defined, the check will ensure constexpr function names will add the
    suffix with the given value (regardless of casing).

For example using values of:

   - ConstexprFunctionCase of ``lower_case``
   - ConstexprFunctionPrefix of ``pre_``
   - ConstexprFunctionSuffix of ``_post``

Identifies and/or transforms constexpr function names as follows:

Before:

.. code-block:: c++

    constexpr int CE_function() { return 3; }

After:

.. code-block:: c++

    constexpr int pre_ce_function_post() { return 3; }

.. option:: ConstexprMethodCase (added in 15.0.0)

    When defined, the check will ensure constexpr method names conform to the
    selected casing.

.. option:: ConstexprMethodPrefix (added in 15.0.0)

    When defined, the check will ensure constexpr method names will add the
    prefixed with the given value (regardless of casing).

.. option:: ConstexprMethodIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for constexpr method names
    matching this regular expression.

.. option:: ConstexprMethodSuffix (added in 15.0.0)

    When defined, the check will ensure constexpr method names will add the
    suffix with the given value (regardless of casing).

For example using values of:

   - ConstexprMethodCase of ``lower_case``
   - ConstexprMethodPrefix of ``pre_``
   - ConstexprMethodSuffix of ``_post``

Identifies and/or transforms constexpr method names as follows:

Before:

.. code-block:: c++

    class Foo {
    public:
      constexpr int CST_expr_Method() { return 2; }
    }

After:

.. code-block:: c++

    class Foo {
    public:
      constexpr int pre_cst_expr_method_post() { return 2; }
    }

.. option:: ConstexprVariableCase (added in 15.0.0)

    When defined, the check will ensure constexpr variable names conform to the
    selected casing.

.. option:: ConstexprVariablePrefix (added in 15.0.0)

    When defined, the check will ensure constexpr variable names will add the
    prefixed with the given value (regardless of casing).

.. option:: ConstexprVariableIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for constexpr variable names
    matching this regular expression.

.. option:: ConstexprVariableSuffix (added in 15.0.0)

    When defined, the check will ensure constexpr variable names will add the
    suffix with the given value (regardless of casing).

.. option:: ConstexprVariableHungarianPrefix (added in 15.0.0)

    When enabled, the check ensures that the declared identifier will
    have a Hungarian notation prefix based on the declared type.

For example using values of:

   - ConstexprVariableCase of ``lower_case``
   - ConstexprVariablePrefix of ``pre_``
   - ConstexprVariableSuffix of ``_post``
   - ConstexprVariableHungarianPrefix of ``On``

Identifies and/or transforms constexpr variable names as follows:

Before:

.. code-block:: c++

    constexpr int ConstExpr_variable = MyConstant;

After:

.. code-block:: c++

    constexpr int pre_constexpr_variable_post = MyConstant;

.. option:: EnumCase (added in 15.0.0)

    When defined, the check will ensure enumeration names conform to the
    selected casing.

.. option:: EnumPrefix (added in 15.0.0)

    When defined, the check will ensure enumeration names will add the
    prefixed with the given value (regardless of casing).

.. option:: EnumIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for enumeration names
    matching this regular expression.

.. option:: EnumSuffix (added in 15.0.0)

    When defined, the check will ensure enumeration names will add the
    suffix with the given value (regardless of casing).

For example using values of:

   - EnumCase of ``lower_case``
   - EnumPrefix of ``pre_``
   - EnumSuffix of ``_post``

Identifies and/or transforms enumeration names as follows:

Before:

.. code-block:: c++

    enum FOO { One, Two, Three };

After:

.. code-block:: c++

    enum pre_foo_post { One, Two, Three };

.. option:: EnumConstantCase (added in 15.0.0)

    When defined, the check will ensure enumeration constant names conform to the
    selected casing.

.. option:: EnumConstantPrefix (added in 15.0.0)

    When defined, the check will ensure enumeration constant names will add the
    prefixed with the given value (regardless of casing).

.. option:: EnumConstantIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for enumeration constant names
    matching this regular expression.

.. option:: EnumConstantSuffix (added in 15.0.0)

    When defined, the check will ensure enumeration constant names will add the
    suffix with the given value (regardless of casing).

.. option:: EnumConstantHungarianPrefix (added in 15.0.0)

    When enabled, the check ensures that the declared identifier will
    have a Hungarian notation prefix based on the declared type.

For example using values of:

   - EnumConstantCase of ``lower_case``
   - EnumConstantPrefix of ``pre_``
   - EnumConstantSuffix of ``_post``
   - EnumConstantHungarianPrefix of ``On``

Identifies and/or transforms enumeration constant names as follows:

Before:

.. code-block:: c++

    enum FOO { One, Two, Three };

After:

.. code-block:: c++

    enum FOO { pre_One_post, pre_Two_post, pre_Three_post };

.. option:: FunctionCase (added in 15.0.0)

    When defined, the check will ensure function names conform to the
    selected casing.

.. option:: FunctionPrefix (added in 15.0.0)

    When defined, the check will ensure function names will add the
    prefixed with the given value (regardless of casing).

.. option:: FunctionIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for function names
    matching this regular expression.

.. option:: FunctionSuffix (added in 15.0.0)

    When defined, the check will ensure function names will add the
    suffix with the given value (regardless of casing).

For example using values of:

   - FunctionCase of ``lower_case``
   - FunctionPrefix of ``pre_``
   - FunctionSuffix of ``_post``

Identifies and/or transforms function names as follows:

Before:

.. code-block:: c++

    char MY_Function_string();

After:

.. code-block:: c++

    char pre_my_function_string_post();

.. option:: GetConfigPerFile (added in 15.0.0)

    When `true` the check will look for the configuration for where an
    identifier is declared. Useful for when included header files use a
    different style.
    Default value is `true`.

.. option:: GlobalConstantCase (added in 15.0.0)

    When defined, the check will ensure global constant names conform to the
    selected casing.

.. option:: GlobalConstantPrefix (added in 15.0.0)

    When defined, the check will ensure global constant names will add the
    prefixed with the given value (regardless of casing).

.. option:: GlobalConstantIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for global constant names
    matching this regular expression.

.. option:: GlobalConstantSuffix (added in 15.0.0)

    When defined, the check will ensure global constant names will add the
    suffix with the given value (regardless of casing).

.. option:: GlobalConstantHungarianPrefix (added in 15.0.0)

    When enabled, the check ensures that the declared identifier will
    have a Hungarian notation prefix based on the declared type.

For example using values of:

   - GlobalConstantCase of ``lower_case``
   - GlobalConstantPrefix of ``pre_``
   - GlobalConstantSuffix of ``_post``
   - GlobalConstantHungarianPrefix of ``On``

Identifies and/or transforms global constant names as follows:

Before:

.. code-block:: c++

    unsigned const MyConstGlobal_array[] = {1, 2, 3};

After:

.. code-block:: c++

    unsigned const pre_myconstglobal_array_post[] = {1, 2, 3};

.. option:: GlobalConstantPointerCase (added in 15.0.0)

    When defined, the check will ensure global constant pointer names conform to the
    selected casing.

.. option:: GlobalConstantPointerPrefix (added in 15.0.0)

    When defined, the check will ensure global constant pointer names will add the
    prefixed with the given value (regardless of casing).

.. option:: GlobalConstantPointerIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for global constant pointer
    names matching this regular expression.

.. option:: GlobalConstantPointerSuffix (added in 15.0.0)

    When defined, the check will ensure global constant pointer names will add the
    suffix with the given value (regardless of casing).

.. option:: GlobalConstantPointerHungarianPrefix (added in 15.0.0)

    When enabled, the check ensures that the declared identifier will
    have a Hungarian notation prefix based on the declared type.

For example using values of:

   - GlobalConstantPointerCase of ``lower_case``
   - GlobalConstantPointerPrefix of ``pre_``
   - GlobalConstantPointerSuffix of ``_post``
   - GlobalConstantPointerHungarianPrefix of ``On``

Identifies and/or transforms global constant pointer names as follows:

Before:

.. code-block:: c++

    int *const MyConstantGlobalPointer = nullptr;

After:

.. code-block:: c++

    int *const pre_myconstantglobalpointer_post = nullptr;

.. option:: GlobalFunctionCase (added in 15.0.0)

    When defined, the check will ensure global function names conform to the
    selected casing.

.. option:: GlobalFunctionPrefix (added in 15.0.0)

    When defined, the check will ensure global function names will add the
    prefixed with the given value (regardless of casing).

.. option:: GlobalFunctionIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for global function names
    matching this regular expression.

.. option:: GlobalFunctionSuffix (added in 15.0.0)

    When defined, the check will ensure global function names will add the
    suffix with the given value (regardless of casing).

For example using values of:

   - GlobalFunctionCase of ``lower_case``
   - GlobalFunctionPrefix of ``pre_``
   - GlobalFunctionSuffix of ``_post``

Identifies and/or transforms global function names as follows:

Before:

.. code-block:: c++

    void GLOBAL_FUNCTION(int PARAMETER_1, int const CONST_parameter);

After:

.. code-block:: c++

    void pre_global_function_post(int PARAMETER_1, int const CONST_parameter);

.. option:: GlobalPointerCase (added in 15.0.0)

    When defined, the check will ensure global pointer names conform to the
    selected casing.

.. option:: GlobalPointerPrefix (added in 15.0.0)

    When defined, the check will ensure global pointer names will add the
    prefixed with the given value (regardless of casing).

.. option:: GlobalPointerIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for global pointer names
    matching this regular expression.

.. option:: GlobalPointerSuffix (added in 15.0.0)

    When defined, the check will ensure global pointer names will add the
    suffix with the given value (regardless of casing).

.. option:: GlobalPointerHungarianPrefix (added in 15.0.0)

    When enabled, the check ensures that the declared identifier will
    have a Hungarian notation prefix based on the declared type.

For example using values of:

   - GlobalPointerCase of ``lower_case``
   - GlobalPointerPrefix of ``pre_``
   - GlobalPointerSuffix of ``_post``
   - GlobalPointerHungarianPrefix of ``On``

Identifies and/or transforms global pointer names as follows:

Before:

.. code-block:: c++

    int *GLOBAL3;

After:

.. code-block:: c++

    int *pre_global3_post;

.. option:: GlobalVariableCase (added in 15.0.0)

    When defined, the check will ensure global variable names conform to the
    selected casing.

.. option:: GlobalVariablePrefix (added in 15.0.0)

    When defined, the check will ensure global variable names will add the
    prefixed with the given value (regardless of casing).

.. option:: GlobalVariableIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for global variable names
    matching this regular expression.

.. option:: GlobalVariableSuffix (added in 15.0.0)

    When defined, the check will ensure global variable names will add the
    suffix with the given value (regardless of casing).

.. option:: GlobalVariableHungarianPrefix (added in 15.0.0)

    When enabled, the check ensures that the declared identifier will
    have a Hungarian notation prefix based on the declared type.

For example using values of:

   - GlobalVariableCase of ``lower_case``
   - GlobalVariablePrefix of ``pre_``
   - GlobalVariableSuffix of ``_post``
   - GlobalVariableHungarianPrefix of ``On``

Identifies and/or transforms global variable names as follows:

Before:

.. code-block:: c++

    int GLOBAL3;

After:

.. code-block:: c++

    int pre_global3_post;

.. option:: IgnoreMainLikeFunctions (added in 15.0.0)

    When set to `true` functions that have a similar signature to ``main`` or
    ``wmain`` won't enforce checks on the names of their parameters.
    Default value is `false`.

.. option:: InlineNamespaceCase (added in 15.0.0)

    When defined, the check will ensure inline namespaces names conform to the
    selected casing.

.. option:: InlineNamespacePrefix (added in 15.0.0)

    When defined, the check will ensure inline namespaces names will add the
    prefixed with the given value (regardless of casing).

.. option:: InlineNamespaceIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for inline namespaces names
    matching this regular expression.

.. option:: InlineNamespaceSuffix (added in 15.0.0)

    When defined, the check will ensure inline namespaces names will add the
    suffix with the given value (regardless of casing).

For example using values of:

   - InlineNamespaceCase of ``lower_case``
   - InlineNamespacePrefix of ``pre_``
   - InlineNamespaceSuffix of ``_post``

Identifies and/or transforms inline namespaces names as follows:

Before:

.. code-block:: c++

    namespace FOO_NS {
    inline namespace InlineNamespace {
    ...
    }
    } // namespace FOO_NS

After:

.. code-block:: c++

    namespace FOO_NS {
    inline namespace pre_inlinenamespace_post {
    ...
    }
    } // namespace FOO_NS

.. option:: LocalConstantCase (added in 15.0.0)

    When defined, the check will ensure local constant names conform to the
    selected casing.

.. option:: LocalConstantPrefix (added in 15.0.0)

    When defined, the check will ensure local constant names will add the
    prefixed with the given value (regardless of casing).

.. option:: LocalConstantIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for local constant names
    matching this regular expression.

.. option:: LocalConstantSuffix (added in 15.0.0)

    When defined, the check will ensure local constant names will add the
    suffix with the given value (regardless of casing).

.. option:: LocalConstantHungarianPrefix (added in 15.0.0)

    When enabled, the check ensures that the declared identifier will
    have a Hungarian notation prefix based on the declared type.

For example using values of:

   - LocalConstantCase of ``lower_case``
   - LocalConstantPrefix of ``pre_``
   - LocalConstantSuffix of ``_post``
   - LocalConstantHungarianPrefix of ``On``

Identifies and/or transforms local constant names as follows:

Before:

.. code-block:: c++

    void foo() { int const local_Constant = 3; }

After:

.. code-block:: c++

    void foo() { int const pre_local_constant_post = 3; }

.. option:: LocalConstantPointerCase (added in 15.0.0)

    When defined, the check will ensure local constant pointer names conform to the
    selected casing.

.. option:: LocalConstantPointerPrefix (added in 15.0.0)

    When defined, the check will ensure local constant pointer names will add the
    prefixed with the given value (regardless of casing).

.. option:: LocalConstantPointerIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for local constant pointer names
    matching this regular expression.

.. option:: LocalConstantPointerSuffix (added in 15.0.0)

    When defined, the check will ensure local constant pointer names will add the
    suffix with the given value (regardless of casing).

.. option:: LocalConstantPointerHungarianPrefix (added in 15.0.0)

    When enabled, the check ensures that the declared identifier will
    have a Hungarian notation prefix based on the declared type.

For example using values of:

   - LocalConstantPointerCase of ``lower_case``
   - LocalConstantPointerPrefix of ``pre_``
   - LocalConstantPointerSuffix of ``_post``
   - LocalConstantPointerHungarianPrefix of ``On``

Identifies and/or transforms local constant pointer names as follows:

Before:

.. code-block:: c++

    void foo() { int const *local_Constant = 3; }

After:

.. code-block:: c++

    void foo() { int const *pre_local_constant_post = 3; }

.. option:: LocalPointerCase (added in 15.0.0)

    When defined, the check will ensure local pointer names conform to the
    selected casing.

.. option:: LocalPointerPrefix (added in 15.0.0)

    When defined, the check will ensure local pointer names will add the
    prefixed with the given value (regardless of casing).

.. option:: LocalPointerIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for local pointer names
    matching this regular expression.

.. option:: LocalPointerSuffix (added in 15.0.0)

    When defined, the check will ensure local pointer names will add the
    suffix with the given value (regardless of casing).

.. option:: LocalPointerHungarianPrefix (added in 15.0.0)

    When enabled, the check ensures that the declared identifier will
    have a Hungarian notation prefix based on the declared type.

For example using values of:

   - LocalPointerCase of ``lower_case``
   - LocalPointerPrefix of ``pre_``
   - LocalPointerSuffix of ``_post``
   - LocalPointerHungarianPrefix of ``On``

Identifies and/or transforms local pointer names as follows:

Before:

.. code-block:: c++

    void foo() { int *local_Constant; }

After:

.. code-block:: c++

    void foo() { int *pre_local_constant_post; }

.. option:: LocalVariableCase (added in 15.0.0)

    When defined, the check will ensure local variable names conform to the
    selected casing.

.. option:: LocalVariablePrefix (added in 15.0.0)

    When defined, the check will ensure local variable names will add the
    prefixed with the given value (regardless of casing).

.. option:: LocalVariableIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for local variable names
    matching this regular expression.

For example using values of:

   - LocalVariableCase of ``CamelCase``
   - LocalVariableIgnoredRegexp of ``\w{1,2}``

Will exclude variables with a length less than or equal to 2 from the
camel case check applied to other variables.

.. option:: LocalVariableSuffix (added in 15.0.0)

    When defined, the check will ensure local variable names will add the
    suffix with the given value (regardless of casing).

.. option:: LocalVariableHungarianPrefix (added in 15.0.0)

    When enabled, the check ensures that the declared identifier will
    have a Hungarian notation prefix based on the declared type.

For example using values of:

   - LocalVariableCase of ``lower_case``
   - LocalVariablePrefix of ``pre_``
   - LocalVariableSuffix of ``_post``
   - LocalVariableHungarianPrefix of ``On``

Identifies and/or transforms local variable names as follows:

Before:

.. code-block:: c++

    void foo() { int local_Constant; }

After:

.. code-block:: c++

    void foo() { int pre_local_constant_post; }

.. option:: MacroDefinitionCase (added in 15.0.0)

    When defined, the check will ensure macro definitions conform to the
    selected casing.

.. option:: MacroDefinitionPrefix (added in 15.0.0)

    When defined, the check will ensure macro definitions will add the
    prefixed with the given value (regardless of casing).

.. option:: MacroDefinitionIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for macro definitions
    matching this regular expression.

.. option:: MacroDefinitionSuffix (added in 15.0.0)

    When defined, the check will ensure macro definitions will add the
    suffix with the given value (regardless of casing).

For example using values of:

   - MacroDefinitionCase of ``lower_case``
   - MacroDefinitionPrefix of ``pre_``
   - MacroDefinitionSuffix of ``_post``

Identifies and/or transforms macro definitions as follows:

Before:

.. code-block:: c

    #define MY_MacroDefinition

After:

.. code-block:: c

    #define pre_my_macro_definition_post

Note: This will not warn on builtin macros or macros defined on the command line
using the ``-D`` flag.

.. option:: MemberCase (added in 15.0.0)

    When defined, the check will ensure member names conform to the
    selected casing.

.. option:: MemberPrefix (added in 15.0.0)

    When defined, the check will ensure member names will add the
    prefixed with the given value (regardless of casing).

.. option:: MemberIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for member names
    matching this regular expression.

.. option:: MemberSuffix (added in 15.0.0)

    When defined, the check will ensure member names will add the
    suffix with the given value (regardless of casing).

.. option:: MemberHungarianPrefix (added in 15.0.0)

    When enabled, the check ensures that the declared identifier will
    have a Hungarian notation prefix based on the declared type.

For example using values of:

   - MemberCase of ``lower_case``
   - MemberPrefix of ``pre_``
   - MemberSuffix of ``_post``
   - MemberHungarianPrefix of ``On``

Identifies and/or transforms member names as follows:

Before:

.. code-block:: c++

    class Foo {
      char MY_ConstMember_string[4];
    }

After:

.. code-block:: c++

    class Foo {
      char pre_my_constmember_string_post[4];
    }

.. option:: MethodCase (added in 15.0.0)

    When defined, the check will ensure method names conform to the
    selected casing.

.. option:: MethodPrefix (added in 15.0.0)

    When defined, the check will ensure method names will add the
    prefixed with the given value (regardless of casing).

.. option:: MethodIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for method names
    matching this regular expression.

.. option:: MethodSuffix (added in 15.0.0)

    When defined, the check will ensure method names will add the
    suffix with the given value (regardless of casing).

For example using values of:

   - MethodCase of ``lower_case``
   - MethodPrefix of ``pre_``
   - MethodSuffix of ``_post``

Identifies and/or transforms method names as follows:

Before:

.. code-block:: c++

    class Foo {
      char MY_Method_string();
    }

After:

.. code-block:: c++

    class Foo {
      char pre_my_method_string_post();
    }

.. option:: NamespaceCase (added in 15.0.0)

    When defined, the check will ensure namespace names conform to the
    selected casing.

.. option:: NamespacePrefix (added in 15.0.0)

    When defined, the check will ensure namespace names will add the
    prefixed with the given value (regardless of casing).

.. option:: NamespaceIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for namespace names
    matching this regular expression.

.. option:: NamespaceSuffix (added in 15.0.0)

    When defined, the check will ensure namespace names will add the
    suffix with the given value (regardless of casing).

For example using values of:

   - NamespaceCase of ``lower_case``
   - NamespacePrefix of ``pre_``
   - NamespaceSuffix of ``_post``

Identifies and/or transforms namespace names as follows:

Before:

.. code-block:: c++

    namespace FOO_NS {
    ...
    }

After:

.. code-block:: c++

    namespace pre_foo_ns_post {
    ...
    }

.. option:: ParameterCase (added in 15.0.0)

    When defined, the check will ensure parameter names conform to the
    selected casing.

.. option:: ParameterPrefix (added in 15.0.0)

    When defined, the check will ensure parameter names will add the
    prefixed with the given value (regardless of casing).

.. option:: ParameterIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for parameter names
    matching this regular expression.

.. option:: ParameterSuffix (added in 15.0.0)

    When defined, the check will ensure parameter names will add the
    suffix with the given value (regardless of casing).

.. option:: ParameterHungarianPrefix (added in 15.0.0)

    When enabled, the check ensures that the declared identifier will
    have a Hungarian notation prefix based on the declared type.

For example using values of:

   - ParameterCase of ``lower_case``
   - ParameterPrefix of ``pre_``
   - ParameterSuffix of ``_post``
   - ParameterHungarianPrefix of ``On``

Identifies and/or transforms parameter names as follows:

Before:

.. code-block:: c++

    void GLOBAL_FUNCTION(int PARAMETER_1, int const CONST_parameter);

After:

.. code-block:: c++

    void GLOBAL_FUNCTION(int pre_parameter_post, int const CONST_parameter);

.. option:: ParameterPackCase (added in 15.0.0)

    When defined, the check will ensure parameter pack names conform to the
    selected casing.

.. option:: ParameterPackPrefix (added in 15.0.0)

    When defined, the check will ensure parameter pack names will add the
    prefixed with the given value (regardless of casing).

.. option:: ParameterPackIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for parameter pack names
    matching this regular expression.

.. option:: ParameterPackSuffix (added in 15.0.0)

    When defined, the check will ensure parameter pack names will add the
    suffix with the given value (regardless of casing).

For example using values of:

   - ParameterPackCase of ``lower_case``
   - ParameterPackPrefix of ``pre_``
   - ParameterPackSuffix of ``_post``

Identifies and/or transforms parameter pack names as follows:

Before:

.. code-block:: c++

    template <typename... TYPE_parameters> {
      void FUNCTION(int... TYPE_parameters);
    }

After:

.. code-block:: c++

    template <typename... TYPE_parameters> {
      void FUNCTION(int... pre_type_parameters_post);
    }

.. option:: PointerParameterCase (added in 15.0.0)

    When defined, the check will ensure pointer parameter names conform to the
    selected casing.

.. option:: PointerParameterPrefix (added in 15.0.0)

    When defined, the check will ensure pointer parameter names will add the
    prefixed with the given value (regardless of casing).

.. option:: PointerParameterIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for pointer parameter names
    matching this regular expression.

.. option:: PointerParameterSuffix (added in 15.0.0)

    When defined, the check will ensure pointer parameter names will add the
    suffix with the given value (regardless of casing).

.. option:: PointerParameterHungarianPrefix (added in 15.0.0)

    When enabled, the check ensures that the declared identifier will
    have a Hungarian notation prefix based on the declared type.

For example using values of:

   - PointerParameterCase of ``lower_case``
   - PointerParameterPrefix of ``pre_``
   - PointerParameterSuffix of ``_post``
   - PointerParameterHungarianPrefix of ``On``

Identifies and/or transforms pointer parameter names as follows:

Before:

.. code-block:: c++

    void FUNCTION(int *PARAMETER);

After:

.. code-block:: c++

    void FUNCTION(int *pre_parameter_post);

.. option:: PrivateMemberCase (added in 15.0.0)

    When defined, the check will ensure private member names conform to the
    selected casing.

.. option:: PrivateMemberPrefix (added in 15.0.0)

    When defined, the check will ensure private member names will add the
    prefixed with the given value (regardless of casing).

.. option:: PrivateMemberIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for private member names
    matching this regular expression.

.. option:: PrivateMemberSuffix (added in 15.0.0)

    When defined, the check will ensure private member names will add the
    suffix with the given value (regardless of casing).

.. option:: PrivateMemberHungarianPrefix (added in 15.0.0)

    When enabled, the check ensures that the declared identifier will
    have a Hungarian notation prefix based on the declared type.

For example using values of:

   - PrivateMemberCase of ``lower_case``
   - PrivateMemberPrefix of ``pre_``
   - PrivateMemberSuffix of ``_post``
   - PrivateMemberHungarianPrefix of ``On``

Identifies and/or transforms private member names as follows:

Before:

.. code-block:: c++

    class Foo {
    private:
      int Member_Variable;
    }

After:

.. code-block:: c++

    class Foo {
    private:
      int pre_member_variable_post;
    }

.. option:: PrivateMethodCase (added in 15.0.0)

    When defined, the check will ensure private method names conform to the
    selected casing.

.. option:: PrivateMethodPrefix (added in 15.0.0)

    When defined, the check will ensure private method names will add the
    prefixed with the given value (regardless of casing).

.. option:: PrivateMethodIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for private method names
    matching this regular expression.

.. option:: PrivateMethodSuffix (added in 15.0.0)

    When defined, the check will ensure private method names will add the
    suffix with the given value (regardless of casing).

For example using values of:

   - PrivateMethodCase of ``lower_case``
   - PrivateMethodPrefix of ``pre_``
   - PrivateMethodSuffix of ``_post``

Identifies and/or transforms private method names as follows:

Before:

.. code-block:: c++

    class Foo {
    private:
      int Member_Method();
    }

After:

.. code-block:: c++

    class Foo {
    private:
      int pre_member_method_post();
    }

.. option:: ProtectedMemberCase (added in 15.0.0)

    When defined, the check will ensure protected member names conform to the
    selected casing.

.. option:: ProtectedMemberPrefix (added in 15.0.0)

    When defined, the check will ensure protected member names will add the
    prefixed with the given value (regardless of casing).

.. option:: ProtectedMemberIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for protected member names
    matching this regular expression.

.. option:: ProtectedMemberSuffix (added in 15.0.0)

    When defined, the check will ensure protected member names will add the
    suffix with the given value (regardless of casing).

.. option:: ProtectedMemberHungarianPrefix (added in 15.0.0)

    When enabled, the check ensures that the declared identifier will
    have a Hungarian notation prefix based on the declared type.

For example using values of:

   - ProtectedMemberCase of ``lower_case``
   - ProtectedMemberPrefix of ``pre_``
   - ProtectedMemberSuffix of ``_post``
   - ProtectedMemberHungarianPrefix of ``On``

Identifies and/or transforms protected member names as follows:

Before:

.. code-block:: c++

    class Foo {
    protected:
      int Member_Variable;
    }

After:

.. code-block:: c++

    class Foo {
    protected:
      int pre_member_variable_post;
    }

.. option:: ProtectedMethodCase (added in 15.0.0)

    When defined, the check will ensure protected method names conform to the
    selected casing.

.. option:: ProtectedMethodPrefix (added in 15.0.0)

    When defined, the check will ensure protected method names will add the
    prefixed with the given value (regardless of casing).

.. option:: ProtectedMethodIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for protected method names
    matching this regular expression.

.. option:: ProtectedMethodSuffix (added in 15.0.0)

    When defined, the check will ensure protected method names will add the
    suffix with the given value (regardless of casing).

For example using values of:

   - ProtectedMethodCase of ``lower_case``
   - ProtectedMethodPrefix of ``pre_``
   - ProtectedMethodSuffix of ``_post``

Identifies and/or transforms protect method names as follows:

Before:

.. code-block:: c++

    class Foo {
    protected:
      int Member_Method();
    }

After:

.. code-block:: c++

    class Foo {
    protected:
      int pre_member_method_post();
    }

.. option:: PublicMemberCase (added in 15.0.0)

    When defined, the check will ensure public member names conform to the
    selected casing.

.. option:: PublicMemberPrefix (added in 15.0.0)

    When defined, the check will ensure public member names will add the
    prefixed with the given value (regardless of casing).

.. option:: PublicMemberIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for public member names
    matching this regular expression.

.. option:: PublicMemberSuffix (added in 15.0.0)

    When defined, the check will ensure public member names will add the
    suffix with the given value (regardless of casing).

.. option:: PublicMemberHungarianPrefix (added in 15.0.0)

    When enabled, the check ensures that the declared identifier will
    have a Hungarian notation prefix based on the declared type.

For example using values of:

   - PublicMemberCase of ``lower_case``
   - PublicMemberPrefix of ``pre_``
   - PublicMemberSuffix of ``_post``
   - PublicMemberHungarianPrefix of ``On``

Identifies and/or transforms public member names as follows:

Before:

.. code-block:: c++

    class Foo {
    public:
      int Member_Variable;
    }

After:

.. code-block:: c++

    class Foo {
    public:
      int pre_member_variable_post;
    }

.. option:: PublicMethodCase (added in 15.0.0)

    When defined, the check will ensure public method names conform to the
    selected casing.

.. option:: PublicMethodPrefix (added in 15.0.0)

    When defined, the check will ensure public method names will add the
    prefixed with the given value (regardless of casing).

.. option:: PublicMethodIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for public method names
    matching this regular expression.

.. option:: PublicMethodSuffix (added in 15.0.0)

    When defined, the check will ensure public method names will add the
    suffix with the given value (regardless of casing).

For example using values of:

   - PublicMethodCase of ``lower_case``
   - PublicMethodPrefix of ``pre_``
   - PublicMethodSuffix of ``_post``

Identifies and/or transforms public method names as follows:

Before:

.. code-block:: c++

    class Foo {
    public:
      int Member_Method();
    }

After:

.. code-block:: c++

    class Foo {
    public:
      int pre_member_method_post();
    }

.. option:: ScopedEnumConstantCase (added in 15.0.0)

    When defined, the check will ensure scoped enum constant names conform to
    the selected casing.

.. option:: ScopedEnumConstantPrefix (added in 15.0.0)

    When defined, the check will ensure scoped enum constant names will add the
    prefixed with the given value (regardless of casing).

.. option:: ScopedEnumConstantIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for scoped enum constant names
    matching this regular expression.

.. option:: ScopedEnumConstantSuffix (added in 15.0.0)

    When defined, the check will ensure scoped enum constant names will add the
    suffix with the given value (regardless of casing).

.. option:: ScopedEnumConstantHungarianPrefix (added in 15.0.0)

    When enabled, the check ensures that the declared identifier will
    have a Hungarian notation prefix based on the declared type.

For example using values of:

   - ScopedEnumConstantCase of ``lower_case``
   - ScopedEnumConstantPrefix of ``pre_``
   - ScopedEnumConstantSuffix of ``_post``
   - ScopedEnumConstantHungarianPrefix of ``On``

Identifies and/or transforms enumeration constant names as follows:

Before:

.. code-block:: c++

    enum class FOO { One, Two, Three };

After:

.. code-block:: c++

    enum class FOO { pre_One_post, pre_Two_post, pre_Three_post };

.. option:: StaticConstantCase (added in 15.0.0)

    When defined, the check will ensure static constant names conform to the
    selected casing.

.. option:: StaticConstantPrefix (added in 15.0.0)

    When defined, the check will ensure static constant names will add the
    prefixed with the given value (regardless of casing).

.. option:: StaticConstantIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for static constant names
    matching this regular expression.

.. option:: StaticConstantSuffix (added in 15.0.0)

    When defined, the check will ensure static constant names will add the
    suffix with the given value (regardless of casing).

.. option:: StaticConstantHungarianPrefix (added in 15.0.0)

    When enabled, the check ensures that the declared identifier will
    have a Hungarian notation prefix based on the declared type.

For example using values of:

   - StaticConstantCase of ``lower_case``
   - StaticConstantPrefix of ``pre_``
   - StaticConstantSuffix of ``_post``
   - StaticConstantHungarianPrefix of ``On``

Identifies and/or transforms static constant names as follows:

Before:

.. code-block:: c++

    static unsigned const MyConstStatic_array[] = {1, 2, 3};

After:

.. code-block:: c++

    static unsigned const pre_myconststatic_array_post[] = {1, 2, 3};

.. option:: StaticVariableCase (added in 15.0.0)

    When defined, the check will ensure static variable names conform to the
    selected casing.

.. option:: StaticVariablePrefix (added in 15.0.0)

    When defined, the check will ensure static variable names will add the
    prefixed with the given value (regardless of casing).

.. option:: StaticVariableIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for static variable names
    matching this regular expression.

.. option:: StaticVariableSuffix (added in 15.0.0)

    When defined, the check will ensure static variable names will add the
    suffix with the given value (regardless of casing).

.. option:: StaticVariableHungarianPrefix (added in 15.0.0)

    When enabled, the check ensures that the declared identifier will
    have a Hungarian notation prefix based on the declared type.

For example using values of:

   - StaticVariableCase of ``lower_case``
   - StaticVariablePrefix of ``pre_``
   - StaticVariableSuffix of ``_post``
   - StaticVariableHungarianPrefix of ``On``

Identifies and/or transforms static variable names as follows:

Before:

.. code-block:: c++

    static unsigned MyStatic_array[] = {1, 2, 3};

After:

.. code-block:: c++

    static unsigned pre_mystatic_array_post[] = {1, 2, 3};

.. option:: StructCase (added in 15.0.0)

    When defined, the check will ensure struct names conform to the
    selected casing.

.. option:: StructPrefix (added in 15.0.0)

    When defined, the check will ensure struct names will add the
    prefixed with the given value (regardless of casing).

.. option:: StructIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for struct names
    matching this regular expression.

.. option:: StructSuffix (added in 15.0.0)

    When defined, the check will ensure struct names will add the
    suffix with the given value (regardless of casing).

For example using values of:

   - StructCase of ``lower_case``
   - StructPrefix of ``pre_``
   - StructSuffix of ``_post``

Identifies and/or transforms struct names as follows:

Before:

.. code-block:: c++

    struct FOO {
      FOO();
      ~FOO();
    };

After:

.. code-block:: c++

    struct pre_foo_post {
      pre_foo_post();
      ~pre_foo_post();
    };

.. option:: TemplateParameterCase (added in 15.0.0)

    When defined, the check will ensure template parameter names conform to the
    selected casing.

.. option:: TemplateParameterPrefix (added in 15.0.0)

    When defined, the check will ensure template parameter names will add the
    prefixed with the given value (regardless of casing).

.. option:: TemplateParameterIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for template parameter names
    matching this regular expression.

.. option:: TemplateParameterSuffix (added in 15.0.0)

    When defined, the check will ensure template parameter names will add the
    suffix with the given value (regardless of casing).

For example using values of:

   - TemplateParameterCase of ``lower_case``
   - TemplateParameterPrefix of ``pre_``
   - TemplateParameterSuffix of ``_post``

Identifies and/or transforms template parameter names as follows:

Before:

.. code-block:: c++

    template <typename T> class Foo {};

After:

.. code-block:: c++

    template <typename pre_t_post> class Foo {};

.. option:: TemplateTemplateParameterCase (added in 15.0.0)

    When defined, the check will ensure template template parameter names conform to the
    selected casing.

.. option:: TemplateTemplateParameterPrefix (added in 15.0.0)

    When defined, the check will ensure template template parameter names will add the
    prefixed with the given value (regardless of casing).

.. option:: TemplateTemplateParameterIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for template template parameter
    names matching this regular expression.

.. option:: TemplateTemplateParameterSuffix (added in 15.0.0)

    When defined, the check will ensure template template parameter names will add the
    suffix with the given value (regardless of casing).

For example using values of:

   - TemplateTemplateParameterCase of ``lower_case``
   - TemplateTemplateParameterPrefix of ``pre_``
   - TemplateTemplateParameterSuffix of ``_post``

Identifies and/or transforms template template parameter names as follows:

Before:

.. code-block:: c++

    template <template <typename> class TPL_parameter, int COUNT_params,
              typename... TYPE_parameters>

After:

.. code-block:: c++

    template <template <typename> class pre_tpl_parameter_post, int COUNT_params,
              typename... TYPE_parameters>

.. option:: TypeAliasCase (added in 15.0.0)

    When defined, the check will ensure type alias names conform to the
    selected casing.

.. option:: TypeAliasPrefix (added in 15.0.0)

    When defined, the check will ensure type alias names will add the
    prefixed with the given value (regardless of casing).

.. option:: TypeAliasIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for type alias names
    matching this regular expression.

.. option:: TypeAliasSuffix (added in 15.0.0)

    When defined, the check will ensure type alias names will add the
    suffix with the given value (regardless of casing).

For example using values of:

   - TypeAliasCase of ``lower_case``
   - TypeAliasPrefix of ``pre_``
   - TypeAliasSuffix of ``_post``

Identifies and/or transforms type alias names as follows:

Before:

.. code-block:: c++

    using MY_STRUCT_TYPE = my_structure;

After:

.. code-block:: c++

    using pre_my_struct_type_post = my_structure;

.. option:: TypedefCase (added in 15.0.0)

    When defined, the check will ensure typedef names conform to the
    selected casing.

.. option:: TypedefPrefix (added in 15.0.0)

    When defined, the check will ensure typedef names will add the
    prefixed with the given value (regardless of casing).

.. option:: TypedefIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for typedef names
    matching this regular expression.

.. option:: TypedefSuffix (added in 15.0.0)

    When defined, the check will ensure typedef names will add the
    suffix with the given value (regardless of casing).

For example using values of:

   - TypedefCase of ``lower_case``
   - TypedefPrefix of ``pre_``
   - TypedefSuffix of ``_post``

Identifies and/or transforms typedef names as follows:

Before:

.. code-block:: c++

    typedef int MYINT;

After:

.. code-block:: c++

    typedef int pre_myint_post;

.. option:: TypeTemplateParameterCase (added in 15.0.0)

    When defined, the check will ensure type template parameter names conform to the
    selected casing.

.. option:: TypeTemplateParameterPrefix (added in 15.0.0)

    When defined, the check will ensure type template parameter names will add the
    prefixed with the given value (regardless of casing).

.. option:: TypeTemplateParameterIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for type template names
    matching this regular expression.

.. option:: TypeTemplateParameterSuffix (added in 15.0.0)

    When defined, the check will ensure type template parameter names will add the
    suffix with the given value (regardless of casing).

For example using values of:

   - TypeTemplateParameterCase of ``lower_case``
   - TypeTemplateParameterPrefix of ``pre_``
   - TypeTemplateParameterSuffix of ``_post``

Identifies and/or transforms type template parameter names as follows:

Before:

.. code-block:: c++

    template <template <typename> class TPL_parameter, int COUNT_params,
              typename... TYPE_parameters>

After:

.. code-block:: c++

    template <template <typename> class TPL_parameter, int COUNT_params,
              typename... pre_type_parameters_post>

.. option:: UnionCase (added in 15.0.0)

    When defined, the check will ensure union names conform to the
    selected casing.

.. option:: UnionPrefix (added in 15.0.0)

    When defined, the check will ensure union names will add the
    prefixed with the given value (regardless of casing).

.. option:: UnionIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for union names
    matching this regular expression.

.. option:: UnionSuffix (added in 15.0.0)

    When defined, the check will ensure union names will add the
    suffix with the given value (regardless of casing).

For example using values of:

   - UnionCase of ``lower_case``
   - UnionPrefix of ``pre_``
   - UnionSuffix of ``_post``

Identifies and/or transforms union names as follows:

Before:

.. code-block:: c++

    union FOO {
      int a;
      char b;
    };

After:

.. code-block:: c++

    union pre_foo_post {
      int a;
      char b;
    };

.. option:: ValueTemplateParameterCase (added in 15.0.0)

    When defined, the check will ensure value template parameter names conform to the
    selected casing.

.. option:: ValueTemplateParameterPrefix (added in 15.0.0)

    When defined, the check will ensure value template parameter names will add the
    prefixed with the given value (regardless of casing).

.. option:: ValueTemplateParameterIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for value template parameter
    names matching this regular expression.

.. option:: ValueTemplateParameterSuffix (added in 15.0.0)

    When defined, the check will ensure value template parameter names will add the
    suffix with the given value (regardless of casing).

For example using values of:

   - ValueTemplateParameterCase of ``lower_case``
   - ValueTemplateParameterPrefix of ``pre_``
   - ValueTemplateParameterSuffix of ``_post``

Identifies and/or transforms value template parameter names as follows:

Before:

.. code-block:: c++

    template <template <typename> class TPL_parameter, int COUNT_params,
              typename... TYPE_parameters>

After:

.. code-block:: c++

    template <template <typename> class TPL_parameter, int pre_count_params_post,
              typename... TYPE_parameters>

.. option:: VariableCase (added in 15.0.0)

    When defined, the check will ensure variable names conform to the
    selected casing.

.. option:: VariablePrefix (added in 15.0.0)

    When defined, the check will ensure variable names will add the
    prefixed with the given value (regardless of casing).

.. option:: VariableIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for variable names
    matching this regular expression.

.. option:: VariableSuffix (added in 15.0.0)

    When defined, the check will ensure variable names will add the
    suffix with the given value (regardless of casing).

.. option:: VariableHungarianPrefix (added in 15.0.0)

    When enabled, the check ensures that the declared identifier will
    have a Hungarian notation prefix based on the declared type.

For example using values of:

   - VariableCase of ``lower_case``
   - VariablePrefix of ``pre_``
   - VariableSuffix of ``_post``
   - VariableHungarianPrefix of ``On``

Identifies and/or transforms variable names as follows:

Before:

.. code-block:: c++

    unsigned MyVariable;

After:

.. code-block:: c++

    unsigned pre_myvariable_post;

.. option:: VirtualMethodCase (added in 15.0.0)

    When defined, the check will ensure virtual method names conform to the
    selected casing.

.. option:: VirtualMethodPrefix (added in 15.0.0)

    When defined, the check will ensure virtual method names will add the
    prefixed with the given value (regardless of casing).

.. option:: VirtualMethodIgnoredRegexp (added in 15.0.0)

    Identifier naming checks won't be enforced for virtual method names
    matching this regular expression.

.. option:: VirtualMethodSuffix (added in 15.0.0)

    When defined, the check will ensure virtual method names will add the
    suffix with the given value (regardless of casing).

For example using values of:

   - VirtualMethodCase of ``lower_case``
   - VirtualMethodPrefix of ``pre_``
   - VirtualMethodSuffix of ``_post``

Identifies and/or transforms virtual method names as follows:

Before:

.. code-block:: c++

    class Foo {
    public:
      virtual int MemberFunction();
    }

After:

.. code-block:: c++

    class Foo {
    public:
      virtual int pre_member_function_post();
    }


The default mapping table of Hungarian Notation
-----------------------------------------------

In Hungarian notation, a variable name starts with a group of lower-case
letters which are mnemonics for the type or purpose of that variable, followed
by whatever name the programmer has chosen; this last part is sometimes
distinguished as the given name. The first character of the given name can be
capitalized to separate it from the type indicators (see also CamelCase).
Otherwise the case of this character denotes scope.

The following table is the default mapping table of Hungarian Notation which
maps Decl to its prefix string. You can also have your own style in config file.

================= ============== ====================== ============== ============== ==============
Primitive Type                                                         Microsoft Type
----------------- -------------- ---------------------- -------------- -------------- --------------
    Type          Prefix         Type                   Prefix         Type           Prefix
================= ============== ====================== ============== ============== ==============
int8_t            i8             signed int             si             BOOL           b
int16_t           i16            signed short           ss             BOOLEAN        b
int32_t           i32            signed short int       ssi            BYTE           by
int64_t           i64            signed long long int   slli           CHAR           c
uint8_t           u8             signed long long       sll            UCHAR          uc
uint16_t          u16            signed long int        sli            SHORT          s
uint32_t          u32            signed long            sl             USHORT         us
uint64_t          u64            signed                 s              WORD           w
char8_t           c8             unsigned long long int ulli           DWORD          dw
char16_t          c16            unsigned long long     ull            DWORD32        dw32
char32_t          c32            unsigned long int      uli            DWORD64        dw64
float             f              unsigned long          ul             LONG           l
double            d              unsigned short int     usi            ULONG          ul
char              c              unsigned short         us             ULONG32        ul32
bool              b              unsigned int           ui             ULONG64        ul64
_Bool             b              unsigned char          uc             ULONGLONG      ull
int               i              unsigned               u              HANDLE         h
size_t            n              long long int          lli            INT            i
short             s              long double            ld             INT8           i8
signed            i              long long              ll             INT16          i16
unsigned          u              long int               li             INT32          i32
long              l              long                   l              INT64          i64
long long         ll             ptrdiff_t              p              UINT           ui
unsigned long     ul             void                   *none*         UINT8          u8
long double       ld                                                   UINT16         u16
ptrdiff_t         p                                                    UINT32         u32
wchar_t           wc                                                   UINT64         u64
short int         si                                                   PVOID          p
short             s
================= ============== ====================== ============== ============== ==============

**There are more trivial options for Hungarian Notation:**

**HungarianNotation.General.***
  Options are not belonging to any specific Decl.

**HungarianNotation.CString.***
  Options for NULL-terminated string.

**HungarianNotation.DerivedType.***
 Options for derived types.

**HungarianNotation.PrimitiveType.***
  Options for primitive types.

**HungarianNotation.UserDefinedType.***
  Options for user-defined types.


Options for Hungarian Notation
------------------------------

- :option:`HungarianNotation.General.TreatStructAsClass`

- :option:`HungarianNotation.DerivedType.Array`
- :option:`HungarianNotation.DerivedType.Pointer`
- :option:`HungarianNotation.DerivedType.FunctionPointer`

- :option:`HungarianNotation.CString.CharPointer`
- :option:`HungarianNotation.CString.CharArray`
- :option:`HungarianNotation.CString.WideCharPointer`
- :option:`HungarianNotation.CString.WideCharArray`

- :option:`HungarianNotation.PrimitiveType.*`
- :option:`HungarianNotation.UserDefinedType.*`

.. option:: HungarianNotation.General.TreatStructAsClass (added in 15.0.0)

    When defined, the check will treat naming of struct as a class.
    The default value is `false`.

.. option:: HungarianNotation.DerivedType.Array (added in 15.0.0)

    When defined, the check will ensure variable name will add the prefix with
    the given string. The default prefix is `a`.

.. option:: HungarianNotation.DerivedType.Pointer (added in 15.0.0)

    When defined, the check will ensure variable name will add the prefix with
    the given string. The default prefix is `p`.

.. option:: HungarianNotation.DerivedType.FunctionPointer (added in 15.0.0)

    When defined, the check will ensure variable name will add the prefix with
    the given string. The default prefix is `fn`.


Before:

.. code-block:: c++

    // Array
    int DataArray[2] = {0};

    // Pointer
    void *DataBuffer = NULL;

    // FunctionPointer
    typedef void (*FUNC_PTR)();
    FUNC_PTR FuncPtr = NULL;

After:

.. code-block:: c++

    // Array
    int aDataArray[2] = {0};

    // Pointer
    void *pDataBuffer = NULL;

    // FunctionPointer
    typedef void (*FUNC_PTR)();
    FUNC_PTR fnFuncPtr = NULL;


.. option:: HungarianNotation.CString.CharPointer (added in 17.0.1)

    When defined, the check will ensure variable name will add the prefix with
    the given string. The default prefix is `sz`.

.. option:: HungarianNotation.CString.CharArray (added in 15.0.0)

    When defined, the check will ensure variable name will add the prefix with
    the given string. The default prefix is `sz`.

.. option:: HungarianNotation.CString.WideCharPointer (added in 17.0.1)

    When defined, the check will ensure variable name will add the prefix with
    the given string. The default prefix is `wsz`.

.. option:: HungarianNotation.CString.WideCharArray (added in 15.0.0)

    When defined, the check will ensure variable name will add the prefix with
    the given string. The default prefix is `wsz`.


Before:

.. code-block:: c++

    // CharPointer
    const char *NamePtr = "Name";

    // CharArray
    const char NameArray[] = "Name";

    // WideCharPointer
    const wchar_t *WideNamePtr = L"Name";

    // WideCharArray
    const wchar_t WideNameArray[] = L"Name";

After:

.. code-block:: c++

    // CharPointer
    const char *szNamePtr = "Name";

    // CharArray
    const char szNameArray[] = "Name";

    // WideCharPointer
    const wchar_t *wszWideNamePtr = L"Name";

    // WideCharArray
    const wchar_t wszWideNameArray[] = L"Name";


.. option:: HungarianNotation.PrimitiveType.* (added in 15.0.0)

    When defined, the check will ensure variable name of involved primitive
    types will add the prefix with the given string. The default prefixes are
    defined in the default mapping table.

.. option:: HungarianNotation.UserDefinedType.* (added in 15.0.0)

    When defined, the check will ensure variable name of involved primitive
    types will add the prefix with the given string. The default prefixes are
    defined in the default mapping table.


Before:

.. code-block:: c++

    int8_t   ValueI8      = 0;
    int16_t  ValueI16     = 0;
    int32_t  ValueI32     = 0;
    int64_t  ValueI64     = 0;
    uint8_t  ValueU8      = 0;
    uint16_t ValueU16     = 0;
    uint32_t ValueU32     = 0;
    uint64_t ValueU64     = 0;
    float    ValueFloat   = 0.0;
    double   ValueDouble  = 0.0;
    ULONG    ValueUlong   = 0;
    DWORD    ValueDword   = 0;

After:

.. code-block:: c++

    int8_t   i8ValueI8    = 0;
    int16_t  i16ValueI16  = 0;
    int32_t  i32ValueI32  = 0;
    int64_t  i64ValueI64  = 0;
    uint8_t  u8ValueU8    = 0;
    uint16_t u16ValueU16  = 0;
    uint32_t u32ValueU32  = 0;
    uint64_t u64ValueU64  = 0;
    float    fValueFloat  = 0.0;
    double   dValueDouble = 0.0;
    ULONG    ulValueUlong = 0;
    DWORD    dwValueDword = 0;
