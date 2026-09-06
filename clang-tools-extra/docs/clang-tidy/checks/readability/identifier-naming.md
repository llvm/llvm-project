```{title} clang-tidy - readability-identifier-naming
```

# readability-identifier-naming

Checks for identifiers naming style mismatch.

This check will try to enforce coding guidelines on the identifiers naming. It
supports one of the following casing types and tries to convert from one to
another if a mismatch is detected

Casing types include:

- `lower_case`
- `UPPER_CASE`
- `camelBack`
- `CamelCase`
- `camel_Snake_Back`
- `Camel_Snake_Case`
- `aNy_CasE`
- `Leading_upper_snake_case`

It also supports a fixed prefix and suffix that will be prepended or appended
to the identifiers, regardless of the casing.

Many configuration options are available, in order to be able to create
different rules for different kinds of identifiers. In general, the rules are
falling back to a more generic rule if the specific case is not configured.

The naming of virtual methods is reported where they occur in the base class,
but not where they are overridden, as it can't be fixed locally there.
This also applies for pseudo-override patterns like CRTP.

`Leading_upper_snake_case` is a naming convention where the first word is
capitalized followed by lower case word(s) separated by underscore(s) '\_'.
Examples include: `Cap_snake_case`, `Cobra_case`, `Foo_bar_baz`,
and `Master_copy_8gb`.

Hungarian notation can be customized using different *HungarianPrefix*
settings. The options and their corresponding values are:

- `Off` - the default setting
- `On` - example: `int iVariable`
- `LowerCase` - example: `int i_Variable`
- `CamelCase` - example: `int IVariable`

The check only enforces style on kinds of identifiers which have been
configured, so an empty config effectively disables it.
The {option}`DefaultCase` option can be used to enforce style on all kinds of
identifiers, then optionally overriden for specific kinds which are desired
with a different case.

For example using values of:

- {option}`DefaultCase` of `lower_case`
- {option}`MacroDefinitionCase` of `UPPER_CASE`
- {option}`TemplateParameterCase` of `CamelCase`

Identifies and transforms names as follows:

Before:

```c++
#define macroDefinition
template <typename typenameParameter>
int functionDeclaration(typenameParameter paramVal, int paramCount);
```

After:

```c++
#define MACRO_DEFINITION
template <typename TypenameParameter>
int function_declarations(TypenameParameter param_val, int param_count);
```

## Options summary

The available options are summarized below:

**General options**

- {option}`AggressiveDependentMemberLookup`
- {option}`AllowTrailingUnderscore`
- {option}`CheckAnonFieldInParent`
- {option}`GetConfigPerFile`
- {option}`IgnoreMainLikeFunctions`
- {option}`TypedefInheritAnonTagConfig`

**Specific options**

- {option}`DefaultCase`, {option}`DefaultPrefix`,
  {option}`DefaultSuffix`, {option}`DefaultIgnoredRegexp`,
  {option}`DefaultHungarianPrefix`
- {option}`AbstractClassCase`, {option}`AbstractClassPrefix`,
  {option}`AbstractClassSuffix`, {option}`AbstractClassIgnoredRegexp`,
  {option}`AbstractClassHungarianPrefix`
- {option}`ClassCase`, {option}`ClassPrefix`, {option}`ClassSuffix`,
  {option}`ClassIgnoredRegexp`, {option}`ClassHungarianPrefix`
- {option}`ClassConstexprCase`, {option}`ClassConstexprPrefix`,
  {option}`ClassConstexprSuffix`, {option}`ClassConstexprIgnoredRegexp`,
  {option}`ClassConstexprHungarianPrefix`
- {option}`ClassConstantCase`, {option}`ClassConstantPrefix`,
  {option}`ClassConstantSuffix`, {option}`ClassConstantIgnoredRegexp`,
  {option}`ClassConstantHungarianPrefix`
- {option}`ClassMemberCase`, {option}`ClassMemberPrefix`,
  {option}`ClassMemberSuffix`, {option}`ClassMemberIgnoredRegexp`,
  {option}`ClassMemberHungarianPrefix`
- {option}`ClassMethodCase`, {option}`ClassMethodPrefix`,
  {option}`ClassMethodSuffix`, {option}`ClassMethodIgnoredRegexp`
- {option}`ConceptCase`, {option}`ConceptPrefix`, {option}`ConceptSuffix`,
  {option}`ConceptIgnoredRegexp`
- {option}`ConstantCase`, {option}`ConstantPrefix`, {option}`ConstantSuffix`,
  {option}`ConstantIgnoredRegexp`, {option}`ConstantHungarianPrefix`
- {option}`ConstantMemberCase`, {option}`ConstantMemberPrefix`,
  {option}`ConstantMemberSuffix`, {option}`ConstantMemberIgnoredRegexp`,
  {option}`ConstantMemberHungarianPrefix`
- {option}`ConstantParameterCase`, {option}`ConstantParameterPrefix`,
  {option}`ConstantParameterSuffix`, {option}`ConstantParameterIgnoredRegexp`,
  {option}`ConstantParameterHungarianPrefix`
- {option}`ConstantPointerParameterCase`,
  {option}`ConstantPointerParameterPrefix`,
  {option}`ConstantPointerParameterSuffix`,
  {option}`ConstantPointerParameterIgnoredRegexp`,
  {option}`ConstantPointerParameterHungarianPrefix`
- {option}`ConstexprFunctionCase`, {option}`ConstexprFunctionPrefix`,
  {option}`ConstexprFunctionSuffix`, {option}`ConstexprFunctionIgnoredRegexp`
- {option}`ConstexprMethodCase`, {option}`ConstexprMethodPrefix`,
  {option}`ConstexprMethodSuffix`, {option}`ConstexprMethodIgnoredRegexp`
- {option}`ConstexprVariableCase`, {option}`ConstexprVariablePrefix`,
  {option}`ConstexprVariableSuffix`, {option}`ConstexprVariableIgnoredRegexp`,
  {option}`ConstexprVariableHungarianPrefix`
- {option}`EnumCase`, {option}`EnumPrefix`, {option}`EnumSuffix`,
  {option}`EnumIgnoredRegexp`
- {option}`EnumConstantCase`, {option}`EnumConstantPrefix`,
  {option}`EnumConstantSuffix`, {option}`EnumConstantIgnoredRegexp`,
  {option}`EnumConstantHungarianPrefix`
- {option}`FunctionCase`, {option}`FunctionPrefix`, {option}`FunctionSuffix`,
  {option}`FunctionIgnoredRegexp`
- {option}`GlobalConstexprVariableCase`,
  {option}`GlobalConstexprVariablePrefix`,
  {option}`GlobalConstexprVariableSuffix`,
  {option}`GlobalConstexprVariableIgnoredRegexp`,
  {option}`GlobalConstexprVariableHungarianPrefix`
- {option}`GlobalConstantCase`, {option}`GlobalConstantPrefix`,
  {option}`GlobalConstantSuffix`, {option}`GlobalConstantIgnoredRegexp`,
  {option}`GlobalConstantHungarianPrefix`
- {option}`GlobalConstantPointerCase`,
  {option}`GlobalConstantPointerPrefix`,
  {option}`GlobalConstantPointerSuffix`,
  {option}`GlobalConstantPointerIgnoredRegexp`,
  {option}`GlobalConstantPointerHungarianPrefix`
- {option}`GlobalFunctionCase`, {option}`GlobalFunctionPrefix`,
  {option}`GlobalFunctionSuffix`, {option}`GlobalFunctionIgnoredRegexp`
- {option}`GlobalPointerCase`, {option}`GlobalPointerPrefix`,
  {option}`GlobalPointerSuffix`, {option}`GlobalPointerIgnoredRegexp`,
  {option}`GlobalPointerHungarianPrefix`
- {option}`GlobalVariableCase`, {option}`GlobalVariablePrefix`,
  {option}`GlobalVariableSuffix`, {option}`GlobalVariableIgnoredRegexp`,
  {option}`GlobalVariableHungarianPrefix`
- {option}`InlineNamespaceCase`, {option}`InlineNamespacePrefix`,
  {option}`InlineNamespaceSuffix`, {option}`InlineNamespaceIgnoredRegexp`
- {option}`LambdaCaptureCase`, {option}`LambdaCapturePrefix`,
  {option}`LambdaCaptureSuffix`, {option}`LambdaCaptureIgnoredRegexp`,
  {option}`LambdaCaptureHungarianPrefix`
- {option}`LocalConstexprVariableCase`,
  {option}`LocalConstexprVariablePrefix`,
  {option}`LocalConstexprVariableSuffix`,
  {option}`LocalConstexprVariableIgnoredRegexp`,
  {option}`LocalConstexprVariableHungarianPrefix`
- {option}`LocalConstantCase`, {option}`LocalConstantPrefix`,
  {option}`LocalConstantSuffix`, {option}`LocalConstantIgnoredRegexp`,
  {option}`LocalConstantHungarianPrefix`
- {option}`LocalConstantPointerCase`,
  {option}`LocalConstantPointerPrefix`,
  {option}`LocalConstantPointerSuffix`,
  {option}`LocalConstantPointerIgnoredRegexp`,
  {option}`LocalConstantPointerHungarianPrefix`
- {option}`LocalPointerCase`, {option}`LocalPointerPrefix`,
  {option}`LocalPointerSuffix`, {option}`LocalPointerIgnoredRegexp`,
  {option}`LocalPointerHungarianPrefix`
- {option}`LocalVariableCase`, {option}`LocalVariablePrefix`,
  {option}`LocalVariableSuffix`, {option}`LocalVariableIgnoredRegexp`,
  {option}`LocalVariableHungarianPrefix`
- {option}`MacroDefinitionCase`, {option}`MacroDefinitionPrefix`,
  {option}`MacroDefinitionSuffix`, {option}`MacroDefinitionIgnoredRegexp`
- {option}`MemberCase`, {option}`MemberPrefix`, {option}`MemberSuffix`,
  {option}`MemberIgnoredRegexp`, {option}`MemberHungarianPrefix`
- {option}`MethodCase`, {option}`MethodPrefix`, {option}`MethodSuffix`,
  {option}`MethodIgnoredRegexp`
- {option}`NamespaceCase`, {option}`NamespacePrefix`,
  {option}`NamespaceSuffix`, {option}`NamespaceIgnoredRegexp`
- {option}`ParameterCase`, {option}`ParameterPrefix`,
  {option}`ParameterSuffix`, {option}`ParameterIgnoredRegexp`,
  {option}`ParameterHungarianPrefix`
- {option}`ParameterPackCase`, {option}`ParameterPackPrefix`,
  {option}`ParameterPackSuffix`, {option}`ParameterPackIgnoredRegexp`
- {option}`PointerParameterCase`, {option}`PointerParameterPrefix`,
  {option}`PointerParameterSuffix`, {option}`PointerParameterIgnoredRegexp`,
  {option}`PointerParameterHungarianPrefix`
- {option}`PrivateMemberCase`, {option}`PrivateMemberPrefix`,
  {option}`PrivateMemberSuffix`, {option}`PrivateMemberIgnoredRegexp`,
  {option}`PrivateMemberHungarianPrefix`
- {option}`PrivateMethodCase`, {option}`PrivateMethodPrefix`,
  {option}`PrivateMethodSuffix`, {option}`PrivateMethodIgnoredRegexp`
- {option}`ProtectedMemberCase`, {option}`ProtectedMemberPrefix`,
  {option}`ProtectedMemberSuffix`, {option}`ProtectedMemberIgnoredRegexp`,
  {option}`ProtectedMemberHungarianPrefix`
- {option}`ProtectedMethodCase`, {option}`ProtectedMethodPrefix`,
  {option}`ProtectedMethodSuffix`, {option}`ProtectedMethodIgnoredRegexp`
- {option}`PublicMemberCase`, {option}`PublicMemberPrefix`,
  {option}`PublicMemberSuffix`, {option}`PublicMemberIgnoredRegexp`,
  {option}`PublicMemberHungarianPrefix`
- {option}`PublicMethodCase`, {option}`PublicMethodPrefix`,
  {option}`PublicMethodSuffix`, {option}`PublicMethodIgnoredRegexp`
- {option}`ScopedEnumConstantCase`, {option}`ScopedEnumConstantPrefix`,
  {option}`ScopedEnumConstantSuffix`,
  {option}`ScopedEnumConstantIgnoredRegexp`
- {option}`StaticConstexprVariableCase`,
  {option}`StaticConstexprVariablePrefix`,
  {option}`StaticConstexprVariableSuffix`,
  {option}`StaticConstexprVariableIgnoredRegexp`,
  {option}`StaticConstexprVariableHungarianPrefix`
- {option}`StaticConstantCase`, {option}`StaticConstantPrefix`,
  {option}`StaticConstantSuffix`, {option}`StaticConstantIgnoredRegexp`,
  {option}`StaticConstantHungarianPrefix`
- {option}`StaticVariableCase`, {option}`StaticVariablePrefix`,
  {option}`StaticVariableSuffix`, {option}`StaticVariableIgnoredRegexp`,
  {option}`StaticVariableHungarianPrefix`
- {option}`StructCase`, {option}`StructPrefix`, {option}`StructSuffix`,
  {option}`StructIgnoredRegexp`
- {option}`TemplateParameterCase`, {option}`TemplateParameterPrefix`,
  {option}`TemplateParameterSuffix`, {option}`TemplateParameterIgnoredRegexp`
- {option}`TemplateTemplateParameterCase`,
  {option}`TemplateTemplateParameterPrefix`,
  {option}`TemplateTemplateParameterSuffix`,
  {option}`TemplateTemplateParameterIgnoredRegexp`
- {option}`TypeAliasCase`, {option}`TypeAliasPrefix`,
  {option}`TypeAliasSuffix`, {option}`TypeAliasIgnoredRegexp`
- {option}`TypedefCase`, {option}`TypedefPrefix`, {option}`TypedefSuffix`,
  {option}`TypedefIgnoredRegexp`
- {option}`TypeTemplateParameterCase`,
  {option}`TypeTemplateParameterPrefix`,
  {option}`TypeTemplateParameterSuffix`,
  {option}`TypeTemplateParameterIgnoredRegexp`
- {option}`UnionCase`, {option}`UnionPrefix`, {option}`UnionSuffix`,
  {option}`UnionIgnoredRegexp`
- {option}`ValueTemplateParameterCase`,
  {option}`ValueTemplateParameterPrefix`,
  {option}`ValueTemplateParameterSuffix`,
  {option}`ValueTemplateParameterIgnoredRegexp`
- {option}`VariableCase`, {option}`VariablePrefix`, {option}`VariableSuffix`,
  {option}`VariableIgnoredRegexp`, {option}`VariableHungarianPrefix`
- {option}`VirtualMethodCase`, {option}`VirtualMethodPrefix`,
  {option}`VirtualMethodSuffix`, {option}`VirtualMethodIgnoredRegexp`

## Options description

A detailed description of each option is presented below:

```{option} DefaultCase
When defined, the check will ensure all names by default conform to the
selected casing.
```

```{option} DefaultPrefix
When defined, the check will ensure all names by default will add the
prefix with the given value (regardless of casing).
```

```{option} DefaultIgnoredRegexp
Identifier naming checks won't be enforced for all names by default
matching this regular expression.
```

```{option} DefaultSuffix
When defined, the check will ensure all names by default will add the
suffix with the given value (regardless of casing).
```

```{option} DefaultHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

```{option} AbstractClassCase
When defined, the check will ensure abstract class names conform to the
selected casing.
```

```{option} AbstractClassPrefix
When defined, the check will ensure abstract class names will add the
prefix with the given value (regardless of casing).
```

```{option} AbstractClassIgnoredRegexp
Identifier naming checks won't be enforced for abstract class names
matching this regular expression.
```

```{option} AbstractClassSuffix
When defined, the check will ensure abstract class names will add the
suffix with the given value (regardless of casing).
```

```{option} AbstractClassHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- {option}`AbstractClassCase` of `lower_case`
- {option}`AbstractClassPrefix` of `pre_`
- {option}`AbstractClassSuffix` of `_post`
- {option}`AbstractClassHungarianPrefix` of `On`

Identifies and/or transforms abstract class names as follows:

Before:

```c++
class ABSTRACT_CLASS {
public:
  ABSTRACT_CLASS();
};
```

After:

```c++
class pre_abstract_class_post {
public:
  pre_abstract_class_post();
};
```

```{option} AggressiveDependentMemberLookup
When `true`, the check will look in dependent base classes for dependent
member references that need changing. This can lead to errors with template
specializations. Default is `false`.
```

For example using values of:

- {option}`ClassMemberCase` of `lower_case`

Before:

```c++
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
```

After if {option}`AggressiveDependentMemberLookup` is `false`:

```c++
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
```

After if {option}`AggressiveDependentMemberLookup` is `true`:

```c++
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
```

```{option} AllowTrailingUnderscore
When `true`, a single trailing underscore is allowed on any identifier, in
addition to whatever casing, prefix and suffix are otherwise configured for
its kind.
```

For example using values:

- {option}`AllowTrailingUnderscore` is `true`
- {option}`LocalVariableCase` is `camelBack`

Transforms names as follows:

Before:

```c++
void f(int value) {
  int Value_ = value;
}
```

After:

```c++
void f(int value) {
  int value_ = value;
}
```

```{option} CheckAnonFieldInParent
When `true`, fields in anonymous records (i.e. anonymous
unions and structs) will be treated as names in the enclosing scope
rather than public members of the anonymous record for the purpose
of name checking.
```

For example:

```c++
class Foo {
private:
  union {
    int iv_;
    float fv_;
  };
};
```

If {option}`CheckAnonFieldInParent` is `false`, you may get warnings
that `iv_` and `fv_` are not coherent to public member names, because
`iv_` and `fv_` are public members of the anonymous union. When
{option}`CheckAnonFieldInParent` is `true`, `iv_` and `fv_` will be
treated as private data members of `Foo` for the purpose of name checking
and thus no warnings will be emitted.

```{option} ClassCase
When defined, the check will ensure class names conform to the
selected casing.
```

```{option} ClassPrefix
When defined, the check will ensure class names will add the
prefix with the given value (regardless of casing).
```

```{option} ClassIgnoredRegexp
Identifier naming checks won't be enforced for class names matching
this regular expression.
```

```{option} ClassSuffix
When defined, the check will ensure class names will add the
suffix with the given value (regardless of casing).
```

```{option} ClassHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- {option}`ClassCase` of `lower_case`
- {option}`ClassPrefix` of `pre_`
- {option}`ClassSuffix` of `_post`
- {option}`ClassHungarianPrefix` of `On`

Identifies and/or transforms class names as follows:

Before:

```c++
class FOO {
public:
  FOO();
  ~FOO();
};
```

After:

```c++
class pre_foo_post {
public:
  pre_foo_post();
  ~pre_foo_post();
};
```

```{option} ClassConstexprCase
When defined, the check will ensure class `constexpr` names conform to
the selected casing.
```

```{option} ClassConstexprPrefix
When defined, the check will ensure class `constexpr` names will add the
prefix with the given value (regardless of casing).
```

```{option} ClassConstexprIgnoredRegexp
Identifier naming checks won't be enforced for class `constexpr` names
matching this regular expression.
```

```{option} ClassConstexprSuffix
When defined, the check will ensure class `constexpr` names will add the
suffix with the given value (regardless of casing).
```

```{option} ClassConstexprHungarianPrefix
When enabled, the check ensures that the declared identifier will have a
Hungarian notation prefix based on the declared type.
```

For example using values of:

- {option}`ClassConstexprCase` of `lower_case`
- {option}`ClassConstexprPrefix` of `pre_`
- {option}`ClassConstexprSuffix` of `_post`
- {option}`ClassConstexprHungarianPrefix` of `On`

Identifies and/or transforms class `constexpr` variable names as follows:

Before:

```c++
class FOO {
public:
  static constexpr int CLASS_CONSTEXPR;
};
```

After:

```c++
class FOO {
public:
  static const int pre_class_constexpr_post;
};
```

```{option} ClassConstantCase
When defined, the check will ensure class constant names conform to the
selected casing.
```

```{option} ClassConstantPrefix
When defined, the check will ensure class constant names will add the
prefix with the given value (regardless of casing).
```

```{option} ClassConstantIgnoredRegexp
Identifier naming checks won't be enforced for class constant names
matching this regular expression.
```

```{option} ClassConstantSuffix
When defined, the check will ensure class constant names will add the
suffix with the given value (regardless of casing).
```

```{option} ClassConstantHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- {option}`ClassConstantCase` of `lower_case`
- {option}`ClassConstantPrefix` of `pre_`
- {option}`ClassConstantSuffix` of `_post`
- {option}`ClassConstantHungarianPrefix` of `On`

Identifies and/or transforms class constant names as follows:

Before:

```c++
class FOO {
public:
  static const int CLASS_CONSTANT;
};
```

After:

```c++
class FOO {
public:
  static const int pre_class_constant_post;
};
```

```{option} ClassMemberCase
When defined, the check will ensure class member names conform to the
selected casing.
```

```{option} ClassMemberPrefix
When defined, the check will ensure class member names will add the
prefix with the given value (regardless of casing).
```

```{option} ClassMemberIgnoredRegexp
Identifier naming checks won't be enforced for class member names
matching this regular expression.
```

```{option} ClassMemberSuffix
When defined, the check will ensure class member names will add the
suffix with the given value (regardless of casing).
```

```{option} ClassMemberHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- {option}`ClassMemberCase` of `lower_case`
- {option}`ClassMemberPrefix` of `pre_`
- {option}`ClassMemberSuffix` of `_post`
- {option}`ClassMemberHungarianPrefix` of `On`

Identifies and/or transforms class member names as follows:

Before:

```c++
class FOO {
public:
  static int CLASS_CONSTANT;
};
```

After:

```c++
class FOO {
public:
  static int pre_class_constant_post;
};
```

```{option} ClassMethodCase
When defined, the check will ensure class method names conform to the
selected casing.
```

```{option} ClassMethodPrefix
When defined, the check will ensure class method names will add the
prefix with the given value (regardless of casing).
```

```{option} ClassMethodIgnoredRegexp
Identifier naming checks won't be enforced for class method names
matching this regular expression.
```

```{option} ClassMethodSuffix
When defined, the check will ensure class method names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- {option}`ClassMethodCase` of `lower_case`
- {option}`ClassMethodPrefix` of `pre_`
- {option}`ClassMethodSuffix` of `_post`

Identifies and/or transforms class method names as follows:

Before:

```c++
class FOO {
public:
  int CLASS_MEMBER();
};
```

After:

```c++
class FOO {
public:
  int pre_class_member_post();
};
```

```{option} ConceptCase
When defined, the check will ensure concept names conform to the
selected casing.
```

```{option} ConceptPrefix
When defined, the check will ensure concept names will add the
prefix with the given value (regardless of casing).
```

```{option} ConceptIgnoredRegexp
Identifier naming checks won't be enforced for concept names
matching this regular expression.
```

```{option} ConceptSuffix
When defined, the check will ensure concept names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- {option}`ConceptCase` of `CamelCase`
- {option}`ConceptPrefix` of `Pre`
- {option}`ConceptSuffix` of `Post`

Identifies and/or transforms concept names as follows:

Before:

```c++
template<typename T> concept my_concept = requires (T t) { {t++}; };
```

After:

```c++
template<typename T> concept PreMyConceptPost = requires (T t) { {t++}; };
```

```{option} ConstantCase
When defined, the check will ensure constant names conform to the
selected casing.
```

```{option} ConstantPrefix
When defined, the check will ensure constant names will add the
prefix with the given value (regardless of casing).
```

```{option} ConstantIgnoredRegexp
Identifier naming checks won't be enforced for constant names
matching this regular expression.
```

```{option} ConstantSuffix
When defined, the check will ensure constant names will add the
suffix with the given value (regardless of casing).
```

```{option} ConstantHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- {option}`ConstantCase` of `lower_case`
- {option}`ConstantPrefix` of `pre_`
- {option}`ConstantSuffix` of `_post`
- {option}`ConstantHungarianPrefix` of `On`

Identifies and/or transforms constant names as follows:

Before:

```c++
void function() { unsigned const MyConst_array[] = {1, 2, 3}; }
```

After:

```c++
void function() { unsigned const pre_myconst_array_post[] = {1, 2, 3}; }
```

```{option} ConstantMemberCase
When defined, the check will ensure constant member names conform to the
selected casing.
```

```{option} ConstantMemberPrefix
When defined, the check will ensure constant member names will add the
prefix with the given value (regardless of casing).
```

```{option} ConstantMemberIgnoredRegexp
Identifier naming checks won't be enforced for constant member names
matching this regular expression.
```

```{option} ConstantMemberSuffix
When defined, the check will ensure constant member names will add the
suffix with the given value (regardless of casing).
```

```{option} ConstantMemberHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- {option}`ConstantMemberCase` of `lower_case`
- {option}`ConstantMemberPrefix` of `pre_`
- {option}`ConstantMemberSuffix` of `_post`
- {option}`ConstantMemberHungarianPrefix` of `On`

Identifies and/or transforms constant member names as follows:

Before:

```c++
class Foo {
  char const MY_ConstMember_string[4] = "123";
}
```

After:

```c++
class Foo {
  char const pre_my_constmember_string_post[4] = "123";
}
```

```{option} ConstantParameterCase
When defined, the check will ensure constant parameter names conform to the
selected casing.
```

```{option} ConstantParameterPrefix
When defined, the check will ensure constant parameter names will add the
prefix with the given value (regardless of casing).
```

```{option} ConstantParameterIgnoredRegexp
Identifier naming checks won't be enforced for constant parameter names
matching this regular expression.
```

```{option} ConstantParameterSuffix
When defined, the check will ensure constant parameter names will add the
suffix with the given value (regardless of casing).
```

```{option} ConstantParameterHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- {option}`ConstantParameterCase` of `lower_case`
- {option}`ConstantParameterPrefix` of `pre_`
- {option}`ConstantParameterSuffix` of `_post`
- {option}`ConstantParameterHungarianPrefix` of `On`

Identifies and/or transforms constant parameter names as follows:

Before:

```c++
void GLOBAL_FUNCTION(int PARAMETER_1, int const CONST_parameter);
```

After:

```c++
void GLOBAL_FUNCTION(int PARAMETER_1, int const pre_const_parameter_post);
```

```{option} ConstantPointerParameterCase
When defined, the check will ensure constant pointer parameter names conform to the
selected casing.
```

```{option} ConstantPointerParameterPrefix
When defined, the check will ensure constant pointer parameter names will add the
prefix with the given value (regardless of casing).
```

```{option} ConstantPointerParameterIgnoredRegexp
Identifier naming checks won't be enforced for constant pointer parameter
names matching this regular expression.
```

```{option} ConstantPointerParameterSuffix
When defined, the check will ensure constant pointer parameter names will add the
suffix with the given value (regardless of casing).
```

```{option} ConstantPointerParameterHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- {option}`ConstantPointerParameterCase` of `lower_case`
- {option}`ConstantPointerParameterPrefix` of `pre_`
- {option}`ConstantPointerParameterSuffix` of `_post`
- {option}`ConstantPointerParameterHungarianPrefix` of `On`

Identifies and/or transforms constant pointer parameter names as follows:

Before:

```c++
void GLOBAL_FUNCTION(int const *CONST_parameter);
```

After:

```c++
void GLOBAL_FUNCTION(int const *pre_const_parameter_post);
```

```{option} ConstexprFunctionCase
When defined, the check will ensure constexpr function names conform to the
selected casing.
```

```{option} ConstexprFunctionPrefix
When defined, the check will ensure constexpr function names will add the
prefix with the given value (regardless of casing).
```

```{option} ConstexprFunctionIgnoredRegexp
Identifier naming checks won't be enforced for constexpr function names
matching this regular expression.
```

```{option} ConstexprFunctionSuffix
When defined, the check will ensure constexpr function names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- {option}`ConstexprFunctionCase` of `lower_case`
- {option}`ConstexprFunctionPrefix` of `pre_`
- {option}`ConstexprFunctionSuffix` of `_post`

Identifies and/or transforms constexpr function names as follows:

Before:

```c++
constexpr int CE_function() { return 3; }
```

After:

```c++
constexpr int pre_ce_function_post() { return 3; }
```

```{option} ConstexprMethodCase
When defined, the check will ensure constexpr method names conform to the
selected casing.
```

```{option} ConstexprMethodPrefix
When defined, the check will ensure constexpr method names will add the
prefix with the given value (regardless of casing).
```

```{option} ConstexprMethodIgnoredRegexp
Identifier naming checks won't be enforced for constexpr method names
matching this regular expression.
```

```{option} ConstexprMethodSuffix
When defined, the check will ensure constexpr method names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- {option}`ConstexprMethodCase` of `lower_case`
- {option}`ConstexprMethodPrefix` of `pre_`
- {option}`ConstexprMethodSuffix` of `_post`

Identifies and/or transforms constexpr method names as follows:

Before:

```c++
class Foo {
public:
  constexpr int CST_expr_Method() { return 2; }
}
```

After:

```c++
class Foo {
public:
  constexpr int pre_cst_expr_method_post() { return 2; }
}
```

```{option} ConstexprVariableCase
When defined, the check will ensure constexpr variable names conform to the
selected casing.
```

```{option} ConstexprVariablePrefix
When defined, the check will ensure constexpr variable names will add the
prefix with the given value (regardless of casing).
```

```{option} ConstexprVariableIgnoredRegexp
Identifier naming checks won't be enforced for constexpr variable names
matching this regular expression.
```

```{option} ConstexprVariableSuffix
When defined, the check will ensure constexpr variable names will add the
suffix with the given value (regardless of casing).
```

```{option} ConstexprVariableHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- {option}`ConstexprVariableCase` of `lower_case`
- {option}`ConstexprVariablePrefix` of `pre_`
- {option}`ConstexprVariableSuffix` of `_post`
- {option}`ConstexprVariableHungarianPrefix` of `On`

Identifies and/or transforms constexpr variable names as follows:

Before:

```c++
constexpr int ConstExpr_variable = MyConstant;
```

After:

```c++
constexpr int pre_constexpr_variable_post = MyConstant;
```

```{option} EnumCase
When defined, the check will ensure enumeration names conform to the
selected casing.
```

```{option} EnumPrefix
When defined, the check will ensure enumeration names will add the
prefix with the given value (regardless of casing).
```

```{option} EnumIgnoredRegexp
Identifier naming checks won't be enforced for enumeration names
matching this regular expression.
```

```{option} EnumSuffix
When defined, the check will ensure enumeration names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- {option}`EnumCase` of `lower_case`
- {option}`EnumPrefix` of `pre_`
- {option}`EnumSuffix` of `_post`

Identifies and/or transforms enumeration names as follows:

Before:

```c++
enum FOO { One, Two, Three };
```

After:

```c++
enum pre_foo_post { One, Two, Three };
```

```{option} EnumConstantCase
When defined, the check will ensure enumeration constant names conform to the
selected casing.
```

```{option} EnumConstantPrefix
When defined, the check will ensure enumeration constant names will add the
prefix with the given value (regardless of casing).
```

```{option} EnumConstantIgnoredRegexp
Identifier naming checks won't be enforced for enumeration constant names
matching this regular expression.
```

```{option} EnumConstantSuffix
When defined, the check will ensure enumeration constant names will add the
suffix with the given value (regardless of casing).
```

```{option} EnumConstantHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- {option}`EnumConstantCase` of `lower_case`
- {option}`EnumConstantPrefix` of `pre_`
- {option}`EnumConstantSuffix` of `_post`
- {option}`EnumConstantHungarianPrefix` of `On`

Identifies and/or transforms enumeration constant names as follows:

Before:

```c++
enum FOO { One, Two, Three };
```

After:

```c++
enum FOO { pre_One_post, pre_Two_post, pre_Three_post };
```

```{option} FunctionCase
When defined, the check will ensure function names conform to the
selected casing.
```

```{option} FunctionPrefix
When defined, the check will ensure function names will add the
prefix with the given value (regardless of casing).
```

```{option} FunctionIgnoredRegexp
Identifier naming checks won't be enforced for function names
matching this regular expression.
```

```{option} FunctionSuffix
When defined, the check will ensure function names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- {option}`FunctionCase` of `lower_case`
- {option}`FunctionPrefix` of `pre_`
- {option}`FunctionSuffix` of `_post`

Identifies and/or transforms function names as follows:

Before:

```c++
char MY_Function_string();
```

After:

```c++
char pre_my_function_string_post();
```

```{option} GetConfigPerFile
When `true`, the check will look for the configuration for where an
identifier is declared. Useful for when included header files use a
different style.
Default is `true`.
```

```{option} GlobalConstexprVariableCase
When defined, the check will ensure global `constexpr` variable names
conform to the selected casing.
```

```{option} GlobalConstexprVariablePrefix
When defined, the check will ensure global `constexpr` variable names
will add the prefixed with the given value (regardless of casing).
```

```{option} GlobalConstexprVariableIgnoredRegexp
Identifier naming checks won't be enforced for global `constexpr`
variable names matching this regular expression.
```

```{option} GlobalConstexprVariableSuffix
When defined, the check will ensure global `constexpr` variable names
will add the suffix with the given value (regardless of casing).
```

```{option} GlobalConstexprVariableHungarianPrefix
When enabled, the check ensures that the declared identifier will have a
Hungarian notation prefix based on the declared type.
```

For example using values of:

- {option}`GlobalConstexprVariableCase` of `lower_case`
- {option}`GlobalConstexprVariablePrefix` of `pre_`
- {option}`GlobalConstexprVariableSuffix` of `_post`
- {option}`GlobalConstexprVariableHungarianPrefix` of `On`

Identifies and/or transforms global `constexpr` variable names as follows:

Before:

```c++
constexpr unsigned ImportantValue = 69;
```

After:

```c++
constexpr unsigned pre_important_value_post = 69;
```

```{option} GlobalConstantCase
When defined, the check will ensure global constant names conform to the
selected casing.
```

```{option} GlobalConstantPrefix
When defined, the check will ensure global constant names will add the
prefix with the given value (regardless of casing).
```

```{option} GlobalConstantIgnoredRegexp
Identifier naming checks won't be enforced for global constant names
matching this regular expression.
```

```{option} GlobalConstantSuffix
When defined, the check will ensure global constant names will add the
suffix with the given value (regardless of casing).
```

```{option} GlobalConstantHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- {option}`GlobalConstantCase` of `lower_case`
- {option}`GlobalConstantPrefix` of `pre_`
- {option}`GlobalConstantSuffix` of `_post`
- {option}`GlobalConstantHungarianPrefix` of `On`

Identifies and/or transforms global constant names as follows:

Before:

```c++
unsigned const MyConstGlobal_array[] = {1, 2, 3};
```

After:

```c++
unsigned const pre_myconstglobal_array_post[] = {1, 2, 3};
```

```{option} GlobalConstantPointerCase
When defined, the check will ensure global constant pointer names conform to the
selected casing.
```

```{option} GlobalConstantPointerPrefix
When defined, the check will ensure global constant pointer names will add the
prefix with the given value (regardless of casing).
```

```{option} GlobalConstantPointerIgnoredRegexp
Identifier naming checks won't be enforced for global constant pointer
names matching this regular expression.
```

```{option} GlobalConstantPointerSuffix
When defined, the check will ensure global constant pointer names will add the
suffix with the given value (regardless of casing).
```

```{option} GlobalConstantPointerHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- {option}`GlobalConstantPointerCase` of `lower_case`
- {option}`GlobalConstantPointerPrefix` of `pre_`
- {option}`GlobalConstantPointerSuffix` of `_post`
- {option}`GlobalConstantPointerHungarianPrefix` of `On`

Identifies and/or transforms global constant pointer names as follows:

Before:

```c++
int *const MyConstantGlobalPointer = nullptr;
```

After:

```c++
int *const pre_myconstantglobalpointer_post = nullptr;
```

```{option} GlobalFunctionCase
When defined, the check will ensure global function names conform to the
selected casing.
```

```{option} GlobalFunctionPrefix
When defined, the check will ensure global function names will add the
prefix with the given value (regardless of casing).
```

```{option} GlobalFunctionIgnoredRegexp
Identifier naming checks won't be enforced for global function names
matching this regular expression.
```

```{option} GlobalFunctionSuffix
When defined, the check will ensure global function names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- {option}`GlobalFunctionCase` of `lower_case`
- {option}`GlobalFunctionPrefix` of `pre_`
- {option}`GlobalFunctionSuffix` of `_post`

Identifies and/or transforms global function names as follows:

Before:

```c++
void GLOBAL_FUNCTION(int PARAMETER_1, int const CONST_parameter);
```

After:

```c++
void pre_global_function_post(int PARAMETER_1, int const CONST_parameter);
```

```{option} GlobalPointerCase
When defined, the check will ensure global pointer names conform to the
selected casing.
```

```{option} GlobalPointerPrefix
When defined, the check will ensure global pointer names will add the
prefix with the given value (regardless of casing).
```

```{option} GlobalPointerIgnoredRegexp
Identifier naming checks won't be enforced for global pointer names
matching this regular expression.
```

```{option} GlobalPointerSuffix
When defined, the check will ensure global pointer names will add the
suffix with the given value (regardless of casing).
```

```{option} GlobalPointerHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- {option}`GlobalPointerCase` of `lower_case`
- {option}`GlobalPointerPrefix` of `pre_`
- {option}`GlobalPointerSuffix` of `_post`
- {option}`GlobalPointerHungarianPrefix` of `On`

Identifies and/or transforms global pointer names as follows:

Before:

```c++
int *GLOBAL3;
```

After:

```c++
int *pre_global3_post;
```

```{option} GlobalVariableCase
When defined, the check will ensure global variable names conform to the
selected casing.
```

```{option} GlobalVariablePrefix
When defined, the check will ensure global variable names will add the
prefix with the given value (regardless of casing).
```

```{option} GlobalVariableIgnoredRegexp
Identifier naming checks won't be enforced for global variable names
matching this regular expression.
```

```{option} GlobalVariableSuffix
When defined, the check will ensure global variable names will add the
suffix with the given value (regardless of casing).
```

```{option} GlobalVariableHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- {option}`GlobalVariableCase` of `lower_case`
- {option}`GlobalVariablePrefix` of `pre_`
- {option}`GlobalVariableSuffix` of `_post`
- {option}`GlobalVariableHungarianPrefix` of `On`

Identifies and/or transforms global variable names as follows:

Before:

```c++
int GLOBAL3;
```

After:

```c++
int pre_global3_post;
```

```{option} IgnoreMainLikeFunctions
When `true`, functions that have a similar signature to `main` or
`wmain` won't enforce checks on the names of their parameters.
Default is `false`.
```

```{option} InlineNamespaceCase
When defined, the check will ensure inline namespaces names conform to the
selected casing.
```

```{option} InlineNamespacePrefix
When defined, the check will ensure inline namespaces names will add the
prefix with the given value (regardless of casing).
```

```{option} InlineNamespaceIgnoredRegexp
Identifier naming checks won't be enforced for inline namespaces names
matching this regular expression.
```

```{option} InlineNamespaceSuffix
When defined, the check will ensure inline namespaces names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- {option}`InlineNamespaceCase` of `lower_case`
- {option}`InlineNamespacePrefix` of `pre_`
- {option}`InlineNamespaceSuffix` of `_post`

Identifies and/or transforms inline namespaces names as follows:

Before:

```c++
namespace FOO_NS {
inline namespace InlineNamespace {
...
}
} // namespace FOO_NS
```

After:

```c++
namespace FOO_NS {
inline namespace pre_inlinenamespace_post {
...
}
} // namespace FOO_NS
```

```{option} LambdaCaptureCase
When defined, the check will ensure lambda init-capture names (e.g.
`Captured` in `[Captured = Var]`) conform to the selected casing.
A simple, non-init capture (e.g. `[Var]` or `[&Var]`) refers to the
same declaration as `Var` itself, so it keeps following whichever
naming style applies to `Var`'s own declaration instead.
```

```{option} LambdaCapturePrefix
When defined, the check will ensure lambda init-capture names will add
the prefix with the given value (regardless of casing).
```

```{option} LambdaCaptureIgnoredRegexp
Identifier naming checks won't be enforced for lambda init-capture names
matching this regular expression.
```

```{option} LambdaCaptureSuffix
When defined, the check will ensure lambda init-capture names will add
the suffix with the given value (regardless of casing).
```

```{option} LambdaCaptureHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- {option}`LambdaCaptureCase` of `CamelCase`
- {option}`LambdaCapturePrefix` of `c_`

Identifies and/or transforms lambda init-capture names as follows:

Before:

```c++
void foo() {
  int local_variable = 0;
  auto lambda = [captured_value = local_variable]() {
    return captured_value;
  };
}
```

After:

```c++
void foo() {
  int local_variable = 0;
  auto lambda = [c_CapturedValue = local_variable]() {
    return c_CapturedValue;
  };
}
```

```{option} LocalConstexprVariableCase
When defined, the check will ensure local `constexpr` variable names
conform to the selected casing.
```

```{option} LocalConstexprVariablePrefix
When defined, the check will ensure local `constexpr` variable names will
add the prefixed with the given value (regardless of casing).
```

```{option} LocalConstexprVariableIgnoredRegexp
Identifier naming checks won't be enforced for local `constexpr` variable
names matching this regular expression.
```

```{option} LocalConstexprVariableSuffix
When defined, the check will ensure local `constexpr` variable names will
add the suffix with the given value (regardless of casing).
```

```{option} LocalConstexprVariableHungarianPrefix
When enabled, the check ensures that the declared identifier will have a
Hungarian notation prefix based on the declared type.
```

For example using values of:

- {option}`LocalConstexprVariableCase` of `lower_case`
- {option}`LocalConstexprVariablePrefix` of `pre_`
- {option}`LocalConstexprVariableSuffix` of `_post`
- {option}`LocalConstexprVariableHungarianPrefix` of `On`

Identifies and/or transforms local `constexpr` variable names as follows:

Before:

```c++
void foo() { int const local_Constexpr = 420; }
```

After:

```c++
void foo() { int const pre_local_constexpr_post = 420; }
```

```{option} LocalConstantCase
When defined, the check will ensure local constant names conform to the
selected casing.
```

```{option} LocalConstantPrefix
When defined, the check will ensure local constant names will add the
prefix with the given value (regardless of casing).
```

```{option} LocalConstantIgnoredRegexp
Identifier naming checks won't be enforced for local constant names
matching this regular expression.
```

```{option} LocalConstantSuffix
When defined, the check will ensure local constant names will add the
suffix with the given value (regardless of casing).
```

```{option} LocalConstantHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- {option}`LocalConstantCase` of `lower_case`
- {option}`LocalConstantPrefix` of `pre_`
- {option}`LocalConstantSuffix` of `_post`
- {option}`LocalConstantHungarianPrefix` of `On`

Identifies and/or transforms local constant names as follows:

Before:

```c++
void foo() { int const local_Constant = 3; }
```

After:

```c++
void foo() { int const pre_local_constant_post = 3; }
```

```{option} LocalConstantPointerCase
When defined, the check will ensure local constant pointer names conform to the
selected casing.
```

```{option} LocalConstantPointerPrefix
When defined, the check will ensure local constant pointer names will add the
prefix with the given value (regardless of casing).
```

```{option} LocalConstantPointerIgnoredRegexp
Identifier naming checks won't be enforced for local constant pointer names
matching this regular expression.
```

```{option} LocalConstantPointerSuffix
When defined, the check will ensure local constant pointer names will add the
suffix with the given value (regardless of casing).
```

```{option} LocalConstantPointerHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- {option}`LocalConstantPointerCase` of `lower_case`
- {option}`LocalConstantPointerPrefix` of `pre_`
- {option}`LocalConstantPointerSuffix` of `_post`
- {option}`LocalConstantPointerHungarianPrefix` of `On`

Identifies and/or transforms local constant pointer names as follows:

Before:

```c++
void foo() { int const *local_Constant = 3; }
```

After:

```c++
void foo() { int const *pre_local_constant_post = 3; }
```

```{option} LocalPointerCase
When defined, the check will ensure local pointer names conform to the
selected casing.
```

```{option} LocalPointerPrefix
When defined, the check will ensure local pointer names will add the
prefix with the given value (regardless of casing).
```

```{option} LocalPointerIgnoredRegexp
Identifier naming checks won't be enforced for local pointer names
matching this regular expression.
```

```{option} LocalPointerSuffix
When defined, the check will ensure local pointer names will add the
suffix with the given value (regardless of casing).
```

```{option} LocalPointerHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- {option}`LocalPointerCase` of `lower_case`
- {option}`LocalPointerPrefix` of `pre_`
- {option}`LocalPointerSuffix` of `_post`
- {option}`LocalPointerHungarianPrefix` of `On`

Identifies and/or transforms local pointer names as follows:

Before:

```c++
void foo() { int *local_Constant; }
```

After:

```c++
void foo() { int *pre_local_constant_post; }
```

```{option} LocalVariableCase
When defined, the check will ensure local variable names conform to the
selected casing.
```

```{option} LocalVariablePrefix
When defined, the check will ensure local variable names will add the
prefix with the given value (regardless of casing).
```

```{option} LocalVariableIgnoredRegexp
Identifier naming checks won't be enforced for local variable names
matching this regular expression.
```

For example using values of:

- {option}`LocalVariableCase` of `CamelCase`
- {option}`LocalVariableIgnoredRegexp` of `\w{1,2}`

Will exclude variables with a length less than or equal to 2 from the
camel case check applied to other variables.

```{option} LocalVariableSuffix
When defined, the check will ensure local variable names will add the
suffix with the given value (regardless of casing).
```

```{option} LocalVariableHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- {option}`LocalVariableCase` of `lower_case`
- {option}`LocalVariablePrefix` of `pre_`
- {option}`LocalVariableSuffix` of `_post`
- {option}`LocalVariableHungarianPrefix` of `On`

Identifies and/or transforms local variable names as follows:

Before:

```c++
void foo() { int local_Constant; }
```

After:

```c++
void foo() { int pre_local_constant_post; }
```

```{option} MacroDefinitionCase
When defined, the check will ensure macro definitions conform to the
selected casing.
```

```{option} MacroDefinitionPrefix
When defined, the check will ensure macro definitions will add the
prefix with the given value (regardless of casing).
```

```{option} MacroDefinitionIgnoredRegexp
Identifier naming checks won't be enforced for macro definitions
matching this regular expression.
```

```{option} MacroDefinitionSuffix
When defined, the check will ensure macro definitions will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- {option}`MacroDefinitionCase` of `lower_case`
- {option}`MacroDefinitionPrefix` of `pre_`
- {option}`MacroDefinitionSuffix` of `_post`

Identifies and/or transforms macro definitions as follows:

Before:

```c
#define MY_MacroDefinition
```

After:

```c
#define pre_my_macro_definition_post
```

Note: This will not warn on builtin macros or macros defined on the
command line using the `-D` flag.

```{option} MemberCase
When defined, the check will ensure member names conform to the
selected casing.
```

```{option} MemberPrefix
When defined, the check will ensure member names will add the
prefix with the given value (regardless of casing).
```

```{option} MemberIgnoredRegexp
Identifier naming checks won't be enforced for member names
matching this regular expression.
```

```{option} MemberSuffix
When defined, the check will ensure member names will add the
suffix with the given value (regardless of casing).
```

```{option} MemberHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- {option}`MemberCase` of `lower_case`
- {option}`MemberPrefix` of `pre_`
- {option}`MemberSuffix` of `_post`
- {option}`MemberHungarianPrefix` of `On`

Identifies and/or transforms member names as follows:

Before:

```c++
class Foo {
  char MY_ConstMember_string[4];
}
```

After:

```c++
class Foo {
  char pre_my_constmember_string_post[4];
}
```

```{option} MethodCase
When defined, the check will ensure method names conform to the
selected casing.
```

```{option} MethodPrefix
When defined, the check will ensure method names will add the
prefix with the given value (regardless of casing).
```

```{option} MethodIgnoredRegexp
Identifier naming checks won't be enforced for method names
matching this regular expression.
```

```{option} MethodSuffix
When defined, the check will ensure method names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- {option}`MethodCase` of `lower_case`
- {option}`MethodPrefix` of `pre_`
- {option}`MethodSuffix` of `_post`

Identifies and/or transforms method names as follows:

Before:

```c++
class Foo {
  char MY_Method_string();
}
```

After:

```c++
class Foo {
  char pre_my_method_string_post();
}
```

```{option} NamespaceCase
When defined, the check will ensure namespace names conform to the
selected casing.
```

```{option} NamespacePrefix
When defined, the check will ensure namespace names will add the
prefix with the given value (regardless of casing).
```

```{option} NamespaceIgnoredRegexp
Identifier naming checks won't be enforced for namespace names
matching this regular expression.
```

```{option} NamespaceSuffix
When defined, the check will ensure namespace names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- {option}`NamespaceCase` of `lower_case`
- {option}`NamespacePrefix` of `pre_`
- {option}`NamespaceSuffix` of `_post`

Identifies and/or transforms namespace names as follows:

Before:

```c++
namespace FOO_NS {
...
}
```

After:

```c++
namespace pre_foo_ns_post {
...
}
```

```{option} ParameterCase
When defined, the check will ensure parameter names conform to the
selected casing.
```

```{option} ParameterPrefix
When defined, the check will ensure parameter names will add the
prefix with the given value (regardless of casing).
```

```{option} ParameterIgnoredRegexp
Identifier naming checks won't be enforced for parameter names
matching this regular expression.
```

```{option} ParameterSuffix
When defined, the check will ensure parameter names will add the
suffix with the given value (regardless of casing).
```

```{option} ParameterHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- {option}`ParameterCase` of `lower_case`
- {option}`ParameterPrefix` of `pre_`
- {option}`ParameterSuffix` of `_post`
- {option}`ParameterHungarianPrefix` of `On`

Identifies and/or transforms parameter names as follows:

Before:

```c++
void GLOBAL_FUNCTION(int PARAMETER_1, int const CONST_parameter);
```

After:

```c++
void GLOBAL_FUNCTION(int pre_parameter_post, int const CONST_parameter);
```

```{option} ParameterPackCase
When defined, the check will ensure parameter pack names conform to the
selected casing.
```

```{option} ParameterPackPrefix
When defined, the check will ensure parameter pack names will add the
prefix with the given value (regardless of casing).
```

```{option} ParameterPackIgnoredRegexp
Identifier naming checks won't be enforced for parameter pack names
matching this regular expression.
```

```{option} ParameterPackSuffix
When defined, the check will ensure parameter pack names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- {option}`ParameterPackCase` of `lower_case`
- {option}`ParameterPackPrefix` of `pre_`
- {option}`ParameterPackSuffix` of `_post`

Identifies and/or transforms parameter pack names as follows:

Before:

```c++
template <typename... TYPE_parameters> {
  void FUNCTION(int... TYPE_parameters);
}
```

After:

```c++
template <typename... TYPE_parameters> {
  void FUNCTION(int... pre_type_parameters_post);
}
```

```{option} PointerParameterCase
When defined, the check will ensure pointer parameter names conform to the
selected casing.
```

```{option} PointerParameterPrefix
When defined, the check will ensure pointer parameter names will add the
prefix with the given value (regardless of casing).
```

```{option} PointerParameterIgnoredRegexp
Identifier naming checks won't be enforced for pointer parameter names
matching this regular expression.
```

```{option} PointerParameterSuffix
When defined, the check will ensure pointer parameter names will add the
suffix with the given value (regardless of casing).
```

```{option} PointerParameterHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- {option}`PointerParameterCase` of `lower_case`
- {option}`PointerParameterPrefix` of `pre_`
- {option}`PointerParameterSuffix` of `_post`
- {option}`PointerParameterHungarianPrefix` of `On`

Identifies and/or transforms pointer parameter names as follows:

Before:

```c++
void FUNCTION(int *PARAMETER);
```

After:

```c++
void FUNCTION(int *pre_parameter_post);
```

```{option} PrivateMemberCase
When defined, the check will ensure private member names conform to the
selected casing.
```

```{option} PrivateMemberPrefix
When defined, the check will ensure private member names will add the
prefix with the given value (regardless of casing).
```

```{option} PrivateMemberIgnoredRegexp
Identifier naming checks won't be enforced for private member names
matching this regular expression.
```

```{option} PrivateMemberSuffix
When defined, the check will ensure private member names will add the
suffix with the given value (regardless of casing).
```

```{option} PrivateMemberHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- {option}`PrivateMemberCase` of `lower_case`
- {option}`PrivateMemberPrefix` of `pre_`
- {option}`PrivateMemberSuffix` of `_post`
- {option}`PrivateMemberHungarianPrefix` of `On`

Identifies and/or transforms private member names as follows:

Before:

```c++
class Foo {
private:
  int Member_Variable;
}
```

After:

```c++
class Foo {
private:
  int pre_member_variable_post;
}
```

```{option} PrivateMethodCase
When defined, the check will ensure private method names conform to the
selected casing.
```

```{option} PrivateMethodPrefix
When defined, the check will ensure private method names will add the
prefix with the given value (regardless of casing).
```

```{option} PrivateMethodIgnoredRegexp
Identifier naming checks won't be enforced for private method names
matching this regular expression.
```

```{option} PrivateMethodSuffix
When defined, the check will ensure private method names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- {option}`PrivateMethodCase` of `lower_case`
- {option}`PrivateMethodPrefix` of `pre_`
- {option}`PrivateMethodSuffix` of `_post`

Identifies and/or transforms private method names as follows:

Before:

```c++
class Foo {
private:
  int Member_Method();
}
```

After:

```c++
class Foo {
private:
  int pre_member_method_post();
}
```

```{option} ProtectedMemberCase
When defined, the check will ensure protected member names conform to the
selected casing.
```

```{option} ProtectedMemberPrefix
When defined, the check will ensure protected member names will add the
prefix with the given value (regardless of casing).
```

```{option} ProtectedMemberIgnoredRegexp
Identifier naming checks won't be enforced for protected member names
matching this regular expression.
```

```{option} ProtectedMemberSuffix
When defined, the check will ensure protected member names will add the
suffix with the given value (regardless of casing).
```

```{option} ProtectedMemberHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- {option}`ProtectedMemberCase` of `lower_case`
- {option}`ProtectedMemberPrefix` of `pre_`
- {option}`ProtectedMemberSuffix` of `_post`
- {option}`ProtectedMemberHungarianPrefix` of `On`

Identifies and/or transforms protected member names as follows:

Before:

```c++
class Foo {
protected:
  int Member_Variable;
}
```

After:

```c++
class Foo {
protected:
  int pre_member_variable_post;
}
```

```{option} ProtectedMethodCase
When defined, the check will ensure protected method names conform to the
selected casing.
```

```{option} ProtectedMethodPrefix
When defined, the check will ensure protected method names will add the
prefix with the given value (regardless of casing).
```

```{option} ProtectedMethodIgnoredRegexp
Identifier naming checks won't be enforced for protected method names
matching this regular expression.
```

```{option} ProtectedMethodSuffix
When defined, the check will ensure protected method names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- {option}`ProtectedMethodCase` of `lower_case`
- {option}`ProtectedMethodPrefix` of `pre_`
- {option}`ProtectedMethodSuffix` of `_post`

Identifies and/or transforms protect method names as follows:

Before:

```c++
class Foo {
protected:
  int Member_Method();
}
```

After:

```c++
class Foo {
protected:
  int pre_member_method_post();
}
```

```{option} PublicMemberCase
When defined, the check will ensure public member names conform to the
selected casing.
```

```{option} PublicMemberPrefix
When defined, the check will ensure public member names will add the
prefix with the given value (regardless of casing).
```

```{option} PublicMemberIgnoredRegexp
Identifier naming checks won't be enforced for public member names
matching this regular expression.
```

```{option} PublicMemberSuffix
When defined, the check will ensure public member names will add the
suffix with the given value (regardless of casing).
```

```{option} PublicMemberHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- {option}`PublicMemberCase` of `lower_case`
- {option}`PublicMemberPrefix` of `pre_`
- {option}`PublicMemberSuffix` of `_post`
- {option}`PublicMemberHungarianPrefix` of `On`

Identifies and/or transforms public member names as follows:

Before:

```c++
class Foo {
public:
  int Member_Variable;
}
```

After:

```c++
class Foo {
public:
  int pre_member_variable_post;
}
```

```{option} PublicMethodCase
When defined, the check will ensure public method names conform to the
selected casing.
```

```{option} PublicMethodPrefix
When defined, the check will ensure public method names will add the
prefix with the given value (regardless of casing).
```

```{option} PublicMethodIgnoredRegexp
Identifier naming checks won't be enforced for public method names
matching this regular expression.
```

```{option} PublicMethodSuffix
When defined, the check will ensure public method names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- {option}`PublicMethodCase` of `lower_case`
- {option}`PublicMethodPrefix` of `pre_`
- {option}`PublicMethodSuffix` of `_post`

Identifies and/or transforms public method names as follows:

Before:

```c++
class Foo {
public:
  int Member_Method();
}
```

After:

```c++
class Foo {
public:
  int pre_member_method_post();
}
```

```{option} ScopedEnumConstantCase
When defined, the check will ensure scoped enum constant names conform to
the selected casing.
```

```{option} ScopedEnumConstantPrefix
When defined, the check will ensure scoped enum constant names will add the
prefix with the given value (regardless of casing).
```

```{option} ScopedEnumConstantIgnoredRegexp
Identifier naming checks won't be enforced for scoped enum constant names
matching this regular expression.
```

```{option} ScopedEnumConstantSuffix
When defined, the check will ensure scoped enum constant names will add the
suffix with the given value (regardless of casing).
```

```{option} ScopedEnumConstantHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- {option}`ScopedEnumConstantCase` of `lower_case`
- {option}`ScopedEnumConstantPrefix` of `pre_`
- {option}`ScopedEnumConstantSuffix` of `_post`
- {option}`ScopedEnumConstantHungarianPrefix` of `On`

Identifies and/or transforms enumeration constant names as follows:

Before:

```c++
enum class FOO { One, Two, Three };
```

After:

```c++
enum class FOO { pre_One_post, pre_Two_post, pre_Three_post };
```

```{option} StaticConstexprVariableCase
When defined, the check will ensure static `constexpr` variable names
conform to the selected casing.
```

```{option} StaticConstexprVariablePrefix
When defined, the check will ensure static `constexpr` variable names
will add the prefixed with the given value (regardless of casing).
```

```{option} StaticConstexprVariableIgnoredRegexp
Identifier naming checks won't be enforced for static `constexpr`
variable names matching this regular expression.
```

```{option} StaticConstexprVariableSuffix
When defined, the check will ensure static `constexpr` variable names
will add the suffix with the given value (regardless of casing).
```

```{option} StaticConstexprVariableHungarianPrefix
When enabled, the check ensures that the declared identifier will have a
Hungarian notation prefix based on the declared type.
```

For example using values of:

- {option}`StaticConstexprVariableCase` of `lower_case`
- {option}`StaticConstexprVariablePrefix` of `pre_`
- {option}`StaticConstexprVariableSuffix` of `_post`
- {option}`StaticConstexprVariableHungarianPrefix` of `On`

Identifies and/or transforms static `constexpr` variable names as follows:

Before:

```c++
static unsigned constexpr MyConstexprStatic_array[] = {1, 2, 3};
```

After:

```c++
static unsigned constexpr pre_my_constexpr_static_array_post[] = {1, 2, 3};
```

```{option} StaticConstantCase
When defined, the check will ensure static constant names conform to the
selected casing.
```

```{option} StaticConstantPrefix
When defined, the check will ensure static constant names will add the
prefix with the given value (regardless of casing).
```

```{option} StaticConstantIgnoredRegexp
Identifier naming checks won't be enforced for static constant names
matching this regular expression.
```

```{option} StaticConstantSuffix
When defined, the check will ensure static constant names will add the
suffix with the given value (regardless of casing).
```

```{option} StaticConstantHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- {option}`StaticConstantCase` of `lower_case`
- {option}`StaticConstantPrefix` of `pre_`
- {option}`StaticConstantSuffix` of `_post`
- {option}`StaticConstantHungarianPrefix` of `On`

Identifies and/or transforms static constant names as follows:

Before:

```c++
static unsigned const MyConstStatic_array[] = {1, 2, 3};
```

After:

```c++
static unsigned const pre_myconststatic_array_post[] = {1, 2, 3};
```

```{option} StaticVariableCase
When defined, the check will ensure static variable names conform to the
selected casing.
```

```{option} StaticVariablePrefix
When defined, the check will ensure static variable names will add the
prefix with the given value (regardless of casing).
```

```{option} StaticVariableIgnoredRegexp
Identifier naming checks won't be enforced for static variable names
matching this regular expression.
```

```{option} StaticVariableSuffix
When defined, the check will ensure static variable names will add the
suffix with the given value (regardless of casing).
```

```{option} StaticVariableHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- {option}`StaticVariableCase` of `lower_case`
- {option}`StaticVariablePrefix` of `pre_`
- {option}`StaticVariableSuffix` of `_post`
- {option}`StaticVariableHungarianPrefix` of `On`

Identifies and/or transforms static variable names as follows:

Before:

```c++
static unsigned MyStatic_array[] = {1, 2, 3};
```

After:

```c++
static unsigned pre_mystatic_array_post[] = {1, 2, 3};
```

```{option} StructCase
When defined, the check will ensure struct names conform to the
selected casing.
```

```{option} StructPrefix
When defined, the check will ensure struct names will add the
prefix with the given value (regardless of casing).
```

```{option} StructIgnoredRegexp
Identifier naming checks won't be enforced for struct names
matching this regular expression.
```

```{option} StructSuffix
When defined, the check will ensure struct names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- {option}`StructCase` of `lower_case`
- {option}`StructPrefix` of `pre_`
- {option}`StructSuffix` of `_post`

Identifies and/or transforms struct names as follows:

Before:

```c++
struct FOO {
  FOO();
  ~FOO();
};
```

After:

```c++
struct pre_foo_post {
  pre_foo_post();
  ~pre_foo_post();
};
```

```{option} TemplateParameterCase
When defined, the check will ensure template parameter names conform to the
selected casing.
```

```{option} TemplateParameterPrefix
When defined, the check will ensure template parameter names will add the
prefix with the given value (regardless of casing).
```

```{option} TemplateParameterIgnoredRegexp
Identifier naming checks won't be enforced for template parameter names
matching this regular expression.
```

```{option} TemplateParameterSuffix
When defined, the check will ensure template parameter names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- {option}`TemplateParameterCase` of `lower_case`
- {option}`TemplateParameterPrefix` of `pre_`
- {option}`TemplateParameterSuffix` of `_post`

Identifies and/or transforms template parameter names as follows:

Before:

```c++
template <typename T> class Foo {};
```

After:

```c++
template <typename pre_t_post> class Foo {};
```

```{option} TemplateTemplateParameterCase
When defined, the check will ensure template template parameter names conform to the
selected casing.
```

```{option} TemplateTemplateParameterPrefix
When defined, the check will ensure template template parameter names will add the
prefix with the given value (regardless of casing).
```

```{option} TemplateTemplateParameterIgnoredRegexp
Identifier naming checks won't be enforced for template template parameter
names matching this regular expression.
```

```{option} TemplateTemplateParameterSuffix
When defined, the check will ensure template template parameter names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- {option}`TemplateTemplateParameterCase` of `lower_case`
- {option}`TemplateTemplateParameterPrefix` of `pre_`
- {option}`TemplateTemplateParameterSuffix` of `_post`

Identifies and/or transforms template template parameter names as follows:

Before:

```c++
template <template <typename> class TPL_parameter, int COUNT_params,
          typename... TYPE_parameters>
```

After:

```c++
template <template <typename> class pre_tpl_parameter_post, int COUNT_params,
          typename... TYPE_parameters>
```

```{option} TypeAliasCase
When defined, the check will ensure type alias names conform to the
selected casing.
```

```{option} TypeAliasPrefix
When defined, the check will ensure type alias names will add the
prefix with the given value (regardless of casing).
```

```{option} TypeAliasIgnoredRegexp
Identifier naming checks won't be enforced for type alias names
matching this regular expression.
```

```{option} TypeAliasSuffix
When defined, the check will ensure type alias names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- {option}`TypeAliasCase` of `lower_case`
- {option}`TypeAliasPrefix` of `pre_`
- {option}`TypeAliasSuffix` of `_post`

Identifies and/or transforms type alias names as follows:

Before:

```c++
using MY_STRUCT_TYPE = my_structure;
```

After:

```c++
using pre_my_struct_type_post = my_structure;
```

```{option} TypedefCase
When defined, the check will ensure typedef names conform to the
selected casing.
```

```{option} TypedefPrefix
When defined, the check will ensure typedef names will add the
prefix with the given value (regardless of casing).
```

```{option} TypedefIgnoredRegexp
Identifier naming checks won't be enforced for typedef names
matching this regular expression.
```

```{option} TypedefSuffix
When defined, the check will ensure typedef names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- {option}`TypedefCase` of `lower_case`
- {option}`TypedefPrefix` of `pre_`
- {option}`TypedefSuffix` of `_post`

Identifies and/or transforms typedef names as follows:

Before:

```c++
typedef int MYINT;
```

After:

```c++
typedef int pre_myint_post;
```

```{option} TypedefInheritAnonTagConfig
When `true`, a typedef or type alias that provides the only name of
an otherwise unnamed tag, as in `typedef enum {} MyEnum;`, is checked
against the naming style configured for the kind of that tag
(`AbstractClass`, `Class`, `Enum`, `Struct` or `Union`, i.e.
{option}`EnumCase`, {option}`EnumPrefix`, {option}`EnumSuffix` and
{option}`EnumIgnoredRegexp` for an enum) rather than against the typedef
or type alias style. If that kind configures no case, prefix or suffix,
the typedef or type alias style still applies. Typedefs of named tags, of
other typedefs and of non-tag types are not affected. Default is `false`.
```

For example using values of:

- {option}`TypedefInheritAnonTagConfig` of `true`
- {option}`EnumCase` of `CamelCase`
- {option}`TypedefCase` of `lower_case`

Identifies and/or transforms names as follows:

Before:

```c++
typedef enum { VAL } my_enum;        // The typedef names the enum.
typedef enum Kind { VAL2 } my_kind;  // Kind names the enum.
```

After:

```c++
typedef enum { VAL } MyEnum;
typedef enum Kind { VAL2 } my_kind;
```

```{option} TypeTemplateParameterCase
When defined, the check will ensure type template parameter names conform to the
selected casing.
```

```{option} TypeTemplateParameterPrefix
When defined, the check will ensure type template parameter names will add the
prefix with the given value (regardless of casing).
```

```{option} TypeTemplateParameterIgnoredRegexp
Identifier naming checks won't be enforced for type template names
matching this regular expression.
```

```{option} TypeTemplateParameterSuffix
When defined, the check will ensure type template parameter names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- {option}`TypeTemplateParameterCase` of `lower_case`
- {option}`TypeTemplateParameterPrefix` of `pre_`
- {option}`TypeTemplateParameterSuffix` of `_post`

Identifies and/or transforms type template parameter names as follows:

Before:

```c++
template <template <typename> class TPL_parameter, int COUNT_params,
          typename... TYPE_parameters>
```

After:

```c++
template <template <typename> class TPL_parameter, int COUNT_params,
          typename... pre_type_parameters_post>
```

```{option} UnionCase
When defined, the check will ensure union names conform to the
selected casing.
```

```{option} UnionPrefix
When defined, the check will ensure union names will add the
prefix with the given value (regardless of casing).
```

```{option} UnionIgnoredRegexp
Identifier naming checks won't be enforced for union names
matching this regular expression.
```

```{option} UnionSuffix
When defined, the check will ensure union names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- {option}`UnionCase` of `lower_case`
- {option}`UnionPrefix` of `pre_`
- {option}`UnionSuffix` of `_post`

Identifies and/or transforms union names as follows:

Before:

```c++
union FOO {
  int a;
  char b;
};
```

After:

```c++
union pre_foo_post {
  int a;
  char b;
};
```

```{option} ValueTemplateParameterCase
When defined, the check will ensure value template parameter names conform to the
selected casing.
```

```{option} ValueTemplateParameterPrefix
When defined, the check will ensure value template parameter names will add the
prefix with the given value (regardless of casing).
```

```{option} ValueTemplateParameterIgnoredRegexp
Identifier naming checks won't be enforced for value template parameter
names matching this regular expression.
```

```{option} ValueTemplateParameterSuffix
When defined, the check will ensure value template parameter names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- {option}`ValueTemplateParameterCase` of `lower_case`
- {option}`ValueTemplateParameterPrefix` of `pre_`
- {option}`ValueTemplateParameterSuffix` of `_post`

Identifies and/or transforms value template parameter names as follows:

Before:

```c++
template <template <typename> class TPL_parameter, int COUNT_params,
          typename... TYPE_parameters>
```

After:

```c++
template <template <typename> class TPL_parameter, int pre_count_params_post,
          typename... TYPE_parameters>
```

```{option} VariableCase
When defined, the check will ensure variable names conform to the
selected casing.
```

```{option} VariablePrefix
When defined, the check will ensure variable names will add the
prefix with the given value (regardless of casing).
```

```{option} VariableIgnoredRegexp
Identifier naming checks won't be enforced for variable names
matching this regular expression.
```

```{option} VariableSuffix
When defined, the check will ensure variable names will add the
suffix with the given value (regardless of casing).
```

```{option} VariableHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- {option}`VariableCase` of `lower_case`
- {option}`VariablePrefix` of `pre_`
- {option}`VariableSuffix` of `_post`
- {option}`VariableHungarianPrefix` of `On`

Identifies and/or transforms variable names as follows:

Before:

```c++
unsigned MyVariable;
```

After:

```c++
unsigned pre_myvariable_post;
```

```{option} VirtualMethodCase
When defined, the check will ensure virtual method names conform to the
selected casing.
```

```{option} VirtualMethodPrefix
When defined, the check will ensure virtual method names will add the
prefix with the given value (regardless of casing).
```

```{option} VirtualMethodIgnoredRegexp
Identifier naming checks won't be enforced for virtual method names
matching this regular expression.
```

```{option} VirtualMethodSuffix
When defined, the check will ensure virtual method names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- {option}`VirtualMethodCase` of `lower_case`
- {option}`VirtualMethodPrefix` of `pre_`
- {option}`VirtualMethodSuffix` of `_post`

Identifies and/or transforms virtual method names as follows:

Before:

```c++
class Foo {
public:
  virtual int MemberFunction();
}
```

After:

```c++
class Foo {
public:
  virtual int pre_member_function_post();
}
```

## The default mapping table of Hungarian Notation

In Hungarian notation, a variable name starts with a group of lower-case
letters which are mnemonics for the type or purpose of that variable, followed
by whatever name the programmer has chosen; this last part is sometimes
distinguished as the given name. The first character of the given name can be
capitalized to separate it from the type indicators (see also CamelCase).
Otherwise the case of this character denotes scope.

The following table maps type names to their default Hungarian notation
prefixes. You can define custom mappings in the configuration file.

```{eval-rst}
============== ======== ====================== ======== ============== ========
Primitive Type                                          Microsoft Type
-------------- -------- ---------------------- -------- -------------- --------
    Type       Prefix   Type                   Prefix   Type           Prefix
============== ======== ====================== ======== ============== ========
int8_t         i8       signed int             si       BOOL           b
int16_t        i16      signed short           ss       BOOLEAN        b
int32_t        i32      signed short int       ssi      BYTE           by
int64_t        i64      signed long long int   slli     CHAR           c
uint8_t        u8       signed long long       sll      UCHAR          uc
uint16_t       u16      signed long int        sli      SHORT          s
uint32_t       u32      signed long            sl       USHORT         us
uint64_t       u64      signed                 s        WORD           w
char8_t        c8       unsigned long long int ulli     DWORD          dw
char16_t       c16      unsigned long long     ull      DWORD32        dw32
char32_t       c32      unsigned long int      uli      DWORD64        dw64
float          f        unsigned long          ul       LONG           l
double         d        unsigned short int     usi      ULONG          ul
char           c        unsigned short         us       ULONG32        ul32
bool           b        unsigned int           ui       ULONG64        ul64
_Bool          b        unsigned char          uc       ULONGLONG      ull
int            i        unsigned               u        HANDLE         h
size_t         n        long long int          lli      INT            i
short          s        long double            ld       INT8           i8
signed         i        long long              ll       INT16          i16
unsigned       u        long int               li       INT32          i32
long           l        long                   l        INT64          i64
long long      ll       ptrdiff_t              p        UINT           ui
unsigned long  ul       void                   *none*   UINT8          u8
long double    ld                                       UINT16         u16
ptrdiff_t      p                                        UINT32         u32
wchar_t        wc                                       UINT64         u64
short int      si                                       PVOID          p
short          s
============== ======== ====================== ======== ============== ========
```

**There are more trivial options for Hungarian Notation:**

- **HungarianNotation.General.\***: Options not belonging to any specific
  declaration.
- **HungarianNotation.CString.\***: Options for null-terminated strings.
- **HungarianNotation.DerivedType.\***: Options for derived types.
- **HungarianNotation.PrimitiveType.\***: Options for primitive types.
- **HungarianNotation.UserDefinedType.\***: Options for user-defined types.

## Options for Hungarian Notation

- {option}`HungarianNotation.General.TreatStructAsClass`
- {option}`HungarianNotation.DerivedType.Array`
- {option}`HungarianNotation.DerivedType.Pointer`
- {option}`HungarianNotation.DerivedType.FunctionPointer`
- {option}`HungarianNotation.CString.CharPointer`
- {option}`HungarianNotation.CString.CharArray`
- {option}`HungarianNotation.CString.WideCharPointer`
- {option}`HungarianNotation.CString.WideCharArray`
- {option}`HungarianNotation.PrimitiveType.*`
- {option}`HungarianNotation.UserDefinedType.*`

```{option} HungarianNotation.General.TreatStructAsClass
When defined, the check will treat naming of struct as a class.
Default is `false`.
```

```{option} HungarianNotation.DerivedType.Array
When defined, the check will ensure variable name will add the prefix with
the given string. Default is `a`.
```

```{option} HungarianNotation.DerivedType.Pointer
When defined, the check will ensure variable name will add the prefix with
the given string. Default is `p`.
```

```{option} HungarianNotation.DerivedType.FunctionPointer
When defined, the check will ensure variable name will add the prefix with
the given string. Default is `fn`.
```

Before:

```c++
// Array
int DataArray[2] = {0};

// Pointer
void *DataBuffer = NULL;

// FunctionPointer
typedef void (*FUNC_PTR)();
FUNC_PTR FuncPtr = NULL;
```

After:

```c++
// Array
int aDataArray[2] = {0};

// Pointer
void *pDataBuffer = NULL;

// FunctionPointer
typedef void (*FUNC_PTR)();
FUNC_PTR fnFuncPtr = NULL;
```

```{option} HungarianNotation.CString.CharPointer
When defined, the check will ensure variable name will add the prefix with
the given string. Default is `sz`.
```

```{option} HungarianNotation.CString.CharArray
When defined, the check will ensure variable name will add the prefix with
the given string. Default is `sz`.
```

```{option} HungarianNotation.CString.WideCharPointer
When defined, the check will ensure variable name will add the prefix with
the given string. Default is `wsz`.
```

```{option} HungarianNotation.CString.WideCharArray
When defined, the check will ensure variable name will add the prefix with
the given string. Default is `wsz`.
```

Before:

```c++
// CharPointer
const char *NamePtr = "Name";

// CharArray
const char NameArray[] = "Name";

// WideCharPointer
const wchar_t *WideNamePtr = L"Name";

// WideCharArray
const wchar_t WideNameArray[] = L"Name";
```

After:

```c++
// CharPointer
const char *szNamePtr = "Name";

// CharArray
const char szNameArray[] = "Name";

// WideCharPointer
const wchar_t *wszWideNamePtr = L"Name";

// WideCharArray
const wchar_t wszWideNameArray[] = L"Name";
```

```{option} HungarianNotation.PrimitiveType.*
When defined, the check will ensure variable name of involved primitive
types will add the prefix with the given string. The default prefixes are
defined in the default mapping table.
```

```{option} HungarianNotation.UserDefinedType.*
When defined, the check will ensure variable name of involved user-defined
types will add the prefix with the given string. The default prefixes are
defined in the default mapping table.
```

Before:

```c++
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
```

After:

```c++
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
```
