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
The [`DefaultCase`](#readability-identifier-naming-default-case) option can be used to enforce style on all kinds of
identifiers, then optionally overriden for specific kinds which are desired
with a different case.

For example using values of:

- [`DefaultCase`](#readability-identifier-naming-default-case) of `lower_case`
- [`MacroDefinitionCase`](#readability-identifier-naming-macro-definition-case) of `UPPER_CASE`
- [`TemplateParameterCase`](#readability-identifier-naming-template-parameter-case) of `CamelCase`

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

- [`AggressiveDependentMemberLookup`](#readability-identifier-naming-aggressive-dependent-member-lookup)
- [`AllowTrailingUnderscore`](#readability-identifier-naming-allow-trailing-underscore)
- [`CheckAnonFieldInParent`](#readability-identifier-naming-check-anon-field-in-parent)
- [`GetConfigPerFile`](#readability-identifier-naming-get-config-per-file)
- [`IgnoreMainLikeFunctions`](#readability-identifier-naming-ignore-main-like-functions)
- [`TypedefInheritAnonTagConfig`](#readability-identifier-naming-typedef-inherit-anon-tag-config)

**Specific options**

- [`DefaultCase`](#readability-identifier-naming-default-case), [`DefaultPrefix`](#readability-identifier-naming-default-prefix),
  [`DefaultSuffix`](#readability-identifier-naming-default-suffix), [`DefaultIgnoredRegexp`](#readability-identifier-naming-default-ignored-regexp),
  [`DefaultHungarianPrefix`](#readability-identifier-naming-default-hungarian-prefix)
- [`AbstractClassCase`](#readability-identifier-naming-abstract-class-case), [`AbstractClassPrefix`](#readability-identifier-naming-abstract-class-prefix),
  [`AbstractClassSuffix`](#readability-identifier-naming-abstract-class-suffix), [`AbstractClassIgnoredRegexp`](#readability-identifier-naming-abstract-class-ignored-regexp),
  [`AbstractClassHungarianPrefix`](#readability-identifier-naming-abstract-class-hungarian-prefix)
- [`ClassCase`](#readability-identifier-naming-class-case), [`ClassPrefix`](#readability-identifier-naming-class-prefix), [`ClassSuffix`](#readability-identifier-naming-class-suffix),
  [`ClassIgnoredRegexp`](#readability-identifier-naming-class-ignored-regexp), [`ClassHungarianPrefix`](#readability-identifier-naming-class-hungarian-prefix)
- [`ClassConstexprCase`](#readability-identifier-naming-class-constexpr-case), [`ClassConstexprPrefix`](#readability-identifier-naming-class-constexpr-prefix),
  [`ClassConstexprSuffix`](#readability-identifier-naming-class-constexpr-suffix), [`ClassConstexprIgnoredRegexp`](#readability-identifier-naming-class-constexpr-ignored-regexp),
  [`ClassConstexprHungarianPrefix`](#readability-identifier-naming-class-constexpr-hungarian-prefix)
- [`ClassConstantCase`](#readability-identifier-naming-class-constant-case), [`ClassConstantPrefix`](#readability-identifier-naming-class-constant-prefix),
  [`ClassConstantSuffix`](#readability-identifier-naming-class-constant-suffix), [`ClassConstantIgnoredRegexp`](#readability-identifier-naming-class-constant-ignored-regexp),
  [`ClassConstantHungarianPrefix`](#readability-identifier-naming-class-constant-hungarian-prefix)
- [`ClassMemberCase`](#readability-identifier-naming-class-member-case), [`ClassMemberPrefix`](#readability-identifier-naming-class-member-prefix),
  [`ClassMemberSuffix`](#readability-identifier-naming-class-member-suffix), [`ClassMemberIgnoredRegexp`](#readability-identifier-naming-class-member-ignored-regexp),
  [`ClassMemberHungarianPrefix`](#readability-identifier-naming-class-member-hungarian-prefix)
- [`ClassMethodCase`](#readability-identifier-naming-class-method-case), [`ClassMethodPrefix`](#readability-identifier-naming-class-method-prefix),
  [`ClassMethodSuffix`](#readability-identifier-naming-class-method-suffix), [`ClassMethodIgnoredRegexp`](#readability-identifier-naming-class-method-ignored-regexp)
- [`ConceptCase`](#readability-identifier-naming-concept-case), [`ConceptPrefix`](#readability-identifier-naming-concept-prefix), [`ConceptSuffix`](#readability-identifier-naming-concept-suffix),
  [`ConceptIgnoredRegexp`](#readability-identifier-naming-concept-ignored-regexp)
- [`ConstantCase`](#readability-identifier-naming-constant-case), [`ConstantPrefix`](#readability-identifier-naming-constant-prefix), [`ConstantSuffix`](#readability-identifier-naming-constant-suffix),
  [`ConstantIgnoredRegexp`](#readability-identifier-naming-constant-ignored-regexp), [`ConstantHungarianPrefix`](#readability-identifier-naming-constant-hungarian-prefix)
- [`ConstantMemberCase`](#readability-identifier-naming-constant-member-case), [`ConstantMemberPrefix`](#readability-identifier-naming-constant-member-prefix),
  [`ConstantMemberSuffix`](#readability-identifier-naming-constant-member-suffix), [`ConstantMemberIgnoredRegexp`](#readability-identifier-naming-constant-member-ignored-regexp),
  [`ConstantMemberHungarianPrefix`](#readability-identifier-naming-constant-member-hungarian-prefix)
- [`ConstantParameterCase`](#readability-identifier-naming-constant-parameter-case), [`ConstantParameterPrefix`](#readability-identifier-naming-constant-parameter-prefix),
  [`ConstantParameterSuffix`](#readability-identifier-naming-constant-parameter-suffix), [`ConstantParameterIgnoredRegexp`](#readability-identifier-naming-constant-parameter-ignored-regexp),
  [`ConstantParameterHungarianPrefix`](#readability-identifier-naming-constant-parameter-hungarian-prefix)
- [`ConstantPointerParameterCase`](#readability-identifier-naming-constant-pointer-parameter-case),
  [`ConstantPointerParameterPrefix`](#readability-identifier-naming-constant-pointer-parameter-prefix),
  [`ConstantPointerParameterSuffix`](#readability-identifier-naming-constant-pointer-parameter-suffix),
  [`ConstantPointerParameterIgnoredRegexp`](#readability-identifier-naming-constant-pointer-parameter-ignored-regexp),
  [`ConstantPointerParameterHungarianPrefix`](#readability-identifier-naming-constant-pointer-parameter-hungarian-prefix)
- [`ConstexprFunctionCase`](#readability-identifier-naming-constexpr-function-case), [`ConstexprFunctionPrefix`](#readability-identifier-naming-constexpr-function-prefix),
  [`ConstexprFunctionSuffix`](#readability-identifier-naming-constexpr-function-suffix), [`ConstexprFunctionIgnoredRegexp`](#readability-identifier-naming-constexpr-function-ignored-regexp)
- [`ConstexprMethodCase`](#readability-identifier-naming-constexpr-method-case), [`ConstexprMethodPrefix`](#readability-identifier-naming-constexpr-method-prefix),
  [`ConstexprMethodSuffix`](#readability-identifier-naming-constexpr-method-suffix), [`ConstexprMethodIgnoredRegexp`](#readability-identifier-naming-constexpr-method-ignored-regexp)
- [`ConstexprVariableCase`](#readability-identifier-naming-constexpr-variable-case), [`ConstexprVariablePrefix`](#readability-identifier-naming-constexpr-variable-prefix),
  [`ConstexprVariableSuffix`](#readability-identifier-naming-constexpr-variable-suffix), [`ConstexprVariableIgnoredRegexp`](#readability-identifier-naming-constexpr-variable-ignored-regexp),
  [`ConstexprVariableHungarianPrefix`](#readability-identifier-naming-constexpr-variable-hungarian-prefix)
- [`EnumCase`](#readability-identifier-naming-enum-case), [`EnumPrefix`](#readability-identifier-naming-enum-prefix), [`EnumSuffix`](#readability-identifier-naming-enum-suffix),
  [`EnumIgnoredRegexp`](#readability-identifier-naming-enum-ignored-regexp)
- [`EnumConstantCase`](#readability-identifier-naming-enum-constant-case), [`EnumConstantPrefix`](#readability-identifier-naming-enum-constant-prefix),
  [`EnumConstantSuffix`](#readability-identifier-naming-enum-constant-suffix), [`EnumConstantIgnoredRegexp`](#readability-identifier-naming-enum-constant-ignored-regexp),
  [`EnumConstantHungarianPrefix`](#readability-identifier-naming-enum-constant-hungarian-prefix)
- [`FunctionCase`](#readability-identifier-naming-function-case), [`FunctionPrefix`](#readability-identifier-naming-function-prefix), [`FunctionSuffix`](#readability-identifier-naming-function-suffix),
  [`FunctionIgnoredRegexp`](#readability-identifier-naming-function-ignored-regexp)
- [`GlobalConstexprVariableCase`](#readability-identifier-naming-global-constexpr-variable-case),
  [`GlobalConstexprVariablePrefix`](#readability-identifier-naming-global-constexpr-variable-prefix),
  [`GlobalConstexprVariableSuffix`](#readability-identifier-naming-global-constexpr-variable-suffix),
  [`GlobalConstexprVariableIgnoredRegexp`](#readability-identifier-naming-global-constexpr-variable-ignored-regexp),
  [`GlobalConstexprVariableHungarianPrefix`](#readability-identifier-naming-global-constexpr-variable-hungarian-prefix)
- [`GlobalConstantCase`](#readability-identifier-naming-global-constant-case), [`GlobalConstantPrefix`](#readability-identifier-naming-global-constant-prefix),
  [`GlobalConstantSuffix`](#readability-identifier-naming-global-constant-suffix), [`GlobalConstantIgnoredRegexp`](#readability-identifier-naming-global-constant-ignored-regexp),
  [`GlobalConstantHungarianPrefix`](#readability-identifier-naming-global-constant-hungarian-prefix)
- [`GlobalConstantPointerCase`](#readability-identifier-naming-global-constant-pointer-case),
  [`GlobalConstantPointerPrefix`](#readability-identifier-naming-global-constant-pointer-prefix),
  [`GlobalConstantPointerSuffix`](#readability-identifier-naming-global-constant-pointer-suffix),
  [`GlobalConstantPointerIgnoredRegexp`](#readability-identifier-naming-global-constant-pointer-ignored-regexp),
  [`GlobalConstantPointerHungarianPrefix`](#readability-identifier-naming-global-constant-pointer-hungarian-prefix)
- [`GlobalFunctionCase`](#readability-identifier-naming-global-function-case), [`GlobalFunctionPrefix`](#readability-identifier-naming-global-function-prefix),
  [`GlobalFunctionSuffix`](#readability-identifier-naming-global-function-suffix), [`GlobalFunctionIgnoredRegexp`](#readability-identifier-naming-global-function-ignored-regexp)
- [`GlobalPointerCase`](#readability-identifier-naming-global-pointer-case), [`GlobalPointerPrefix`](#readability-identifier-naming-global-pointer-prefix),
  [`GlobalPointerSuffix`](#readability-identifier-naming-global-pointer-suffix), [`GlobalPointerIgnoredRegexp`](#readability-identifier-naming-global-pointer-ignored-regexp),
  [`GlobalPointerHungarianPrefix`](#readability-identifier-naming-global-pointer-hungarian-prefix)
- [`GlobalVariableCase`](#readability-identifier-naming-global-variable-case), [`GlobalVariablePrefix`](#readability-identifier-naming-global-variable-prefix),
  [`GlobalVariableSuffix`](#readability-identifier-naming-global-variable-suffix), [`GlobalVariableIgnoredRegexp`](#readability-identifier-naming-global-variable-ignored-regexp),
  [`GlobalVariableHungarianPrefix`](#readability-identifier-naming-global-variable-hungarian-prefix)
- [`InlineNamespaceCase`](#readability-identifier-naming-inline-namespace-case), [`InlineNamespacePrefix`](#readability-identifier-naming-inline-namespace-prefix),
  [`InlineNamespaceSuffix`](#readability-identifier-naming-inline-namespace-suffix), [`InlineNamespaceIgnoredRegexp`](#readability-identifier-naming-inline-namespace-ignored-regexp)
- [`LambdaCaptureCase`](#readability-identifier-naming-lambda-capture-case), [`LambdaCapturePrefix`](#readability-identifier-naming-lambda-capture-prefix),
  [`LambdaCaptureSuffix`](#readability-identifier-naming-lambda-capture-suffix), [`LambdaCaptureIgnoredRegexp`](#readability-identifier-naming-lambda-capture-ignored-regexp),
  [`LambdaCaptureHungarianPrefix`](#readability-identifier-naming-lambda-capture-hungarian-prefix)
- [`LocalConstexprVariableCase`](#readability-identifier-naming-local-constexpr-variable-case),
  [`LocalConstexprVariablePrefix`](#readability-identifier-naming-local-constexpr-variable-prefix),
  [`LocalConstexprVariableSuffix`](#readability-identifier-naming-local-constexpr-variable-suffix),
  [`LocalConstexprVariableIgnoredRegexp`](#readability-identifier-naming-local-constexpr-variable-ignored-regexp),
  [`LocalConstexprVariableHungarianPrefix`](#readability-identifier-naming-local-constexpr-variable-hungarian-prefix)
- [`LocalConstantCase`](#readability-identifier-naming-local-constant-case), [`LocalConstantPrefix`](#readability-identifier-naming-local-constant-prefix),
  [`LocalConstantSuffix`](#readability-identifier-naming-local-constant-suffix), [`LocalConstantIgnoredRegexp`](#readability-identifier-naming-local-constant-ignored-regexp),
  [`LocalConstantHungarianPrefix`](#readability-identifier-naming-local-constant-hungarian-prefix)
- [`LocalConstantPointerCase`](#readability-identifier-naming-local-constant-pointer-case),
  [`LocalConstantPointerPrefix`](#readability-identifier-naming-local-constant-pointer-prefix),
  [`LocalConstantPointerSuffix`](#readability-identifier-naming-local-constant-pointer-suffix),
  [`LocalConstantPointerIgnoredRegexp`](#readability-identifier-naming-local-constant-pointer-ignored-regexp),
  [`LocalConstantPointerHungarianPrefix`](#readability-identifier-naming-local-constant-pointer-hungarian-prefix)
- [`LocalPointerCase`](#readability-identifier-naming-local-pointer-case), [`LocalPointerPrefix`](#readability-identifier-naming-local-pointer-prefix),
  [`LocalPointerSuffix`](#readability-identifier-naming-local-pointer-suffix), [`LocalPointerIgnoredRegexp`](#readability-identifier-naming-local-pointer-ignored-regexp),
  [`LocalPointerHungarianPrefix`](#readability-identifier-naming-local-pointer-hungarian-prefix)
- [`LocalVariableCase`](#readability-identifier-naming-local-variable-case), [`LocalVariablePrefix`](#readability-identifier-naming-local-variable-prefix),
  [`LocalVariableSuffix`](#readability-identifier-naming-local-variable-suffix), [`LocalVariableIgnoredRegexp`](#readability-identifier-naming-local-variable-ignored-regexp),
  [`LocalVariableHungarianPrefix`](#readability-identifier-naming-local-variable-hungarian-prefix)
- [`MacroDefinitionCase`](#readability-identifier-naming-macro-definition-case), [`MacroDefinitionPrefix`](#readability-identifier-naming-macro-definition-prefix),
  [`MacroDefinitionSuffix`](#readability-identifier-naming-macro-definition-suffix), [`MacroDefinitionIgnoredRegexp`](#readability-identifier-naming-macro-definition-ignored-regexp)
- [`MemberCase`](#readability-identifier-naming-member-case), [`MemberPrefix`](#readability-identifier-naming-member-prefix), [`MemberSuffix`](#readability-identifier-naming-member-suffix),
  [`MemberIgnoredRegexp`](#readability-identifier-naming-member-ignored-regexp), [`MemberHungarianPrefix`](#readability-identifier-naming-member-hungarian-prefix)
- [`MethodCase`](#readability-identifier-naming-method-case), [`MethodPrefix`](#readability-identifier-naming-method-prefix), [`MethodSuffix`](#readability-identifier-naming-method-suffix),
  [`MethodIgnoredRegexp`](#readability-identifier-naming-method-ignored-regexp)
- [`NamespaceCase`](#readability-identifier-naming-namespace-case), [`NamespacePrefix`](#readability-identifier-naming-namespace-prefix),
  [`NamespaceSuffix`](#readability-identifier-naming-namespace-suffix), [`NamespaceIgnoredRegexp`](#readability-identifier-naming-namespace-ignored-regexp)
- [`ParameterCase`](#readability-identifier-naming-parameter-case), [`ParameterPrefix`](#readability-identifier-naming-parameter-prefix),
  [`ParameterSuffix`](#readability-identifier-naming-parameter-suffix), [`ParameterIgnoredRegexp`](#readability-identifier-naming-parameter-ignored-regexp),
  [`ParameterHungarianPrefix`](#readability-identifier-naming-parameter-hungarian-prefix)
- [`ParameterPackCase`](#readability-identifier-naming-parameter-pack-case), [`ParameterPackPrefix`](#readability-identifier-naming-parameter-pack-prefix),
  [`ParameterPackSuffix`](#readability-identifier-naming-parameter-pack-suffix), [`ParameterPackIgnoredRegexp`](#readability-identifier-naming-parameter-pack-ignored-regexp)
- [`PointerParameterCase`](#readability-identifier-naming-pointer-parameter-case), [`PointerParameterPrefix`](#readability-identifier-naming-pointer-parameter-prefix),
  [`PointerParameterSuffix`](#readability-identifier-naming-pointer-parameter-suffix), [`PointerParameterIgnoredRegexp`](#readability-identifier-naming-pointer-parameter-ignored-regexp),
  [`PointerParameterHungarianPrefix`](#readability-identifier-naming-pointer-parameter-hungarian-prefix)
- [`PrivateMemberCase`](#readability-identifier-naming-private-member-case), [`PrivateMemberPrefix`](#readability-identifier-naming-private-member-prefix),
  [`PrivateMemberSuffix`](#readability-identifier-naming-private-member-suffix), [`PrivateMemberIgnoredRegexp`](#readability-identifier-naming-private-member-ignored-regexp),
  [`PrivateMemberHungarianPrefix`](#readability-identifier-naming-private-member-hungarian-prefix)
- [`PrivateMethodCase`](#readability-identifier-naming-private-method-case), [`PrivateMethodPrefix`](#readability-identifier-naming-private-method-prefix),
  [`PrivateMethodSuffix`](#readability-identifier-naming-private-method-suffix), [`PrivateMethodIgnoredRegexp`](#readability-identifier-naming-private-method-ignored-regexp)
- [`ProtectedMemberCase`](#readability-identifier-naming-protected-member-case), [`ProtectedMemberPrefix`](#readability-identifier-naming-protected-member-prefix),
  [`ProtectedMemberSuffix`](#readability-identifier-naming-protected-member-suffix), [`ProtectedMemberIgnoredRegexp`](#readability-identifier-naming-protected-member-ignored-regexp),
  [`ProtectedMemberHungarianPrefix`](#readability-identifier-naming-protected-member-hungarian-prefix)
- [`ProtectedMethodCase`](#readability-identifier-naming-protected-method-case), [`ProtectedMethodPrefix`](#readability-identifier-naming-protected-method-prefix),
  [`ProtectedMethodSuffix`](#readability-identifier-naming-protected-method-suffix), [`ProtectedMethodIgnoredRegexp`](#readability-identifier-naming-protected-method-ignored-regexp)
- [`PublicMemberCase`](#readability-identifier-naming-public-member-case), [`PublicMemberPrefix`](#readability-identifier-naming-public-member-prefix),
  [`PublicMemberSuffix`](#readability-identifier-naming-public-member-suffix), [`PublicMemberIgnoredRegexp`](#readability-identifier-naming-public-member-ignored-regexp),
  [`PublicMemberHungarianPrefix`](#readability-identifier-naming-public-member-hungarian-prefix)
- [`PublicMethodCase`](#readability-identifier-naming-public-method-case), [`PublicMethodPrefix`](#readability-identifier-naming-public-method-prefix),
  [`PublicMethodSuffix`](#readability-identifier-naming-public-method-suffix), [`PublicMethodIgnoredRegexp`](#readability-identifier-naming-public-method-ignored-regexp)
- [`ScopedEnumConstantCase`](#readability-identifier-naming-scoped-enum-constant-case), [`ScopedEnumConstantPrefix`](#readability-identifier-naming-scoped-enum-constant-prefix),
  [`ScopedEnumConstantSuffix`](#readability-identifier-naming-scoped-enum-constant-suffix),
  [`ScopedEnumConstantIgnoredRegexp`](#readability-identifier-naming-scoped-enum-constant-ignored-regexp)
- [`StaticConstexprVariableCase`](#readability-identifier-naming-static-constexpr-variable-case),
  [`StaticConstexprVariablePrefix`](#readability-identifier-naming-static-constexpr-variable-prefix),
  [`StaticConstexprVariableSuffix`](#readability-identifier-naming-static-constexpr-variable-suffix),
  [`StaticConstexprVariableIgnoredRegexp`](#readability-identifier-naming-static-constexpr-variable-ignored-regexp),
  [`StaticConstexprVariableHungarianPrefix`](#readability-identifier-naming-static-constexpr-variable-hungarian-prefix)
- [`StaticConstantCase`](#readability-identifier-naming-static-constant-case), [`StaticConstantPrefix`](#readability-identifier-naming-static-constant-prefix),
  [`StaticConstantSuffix`](#readability-identifier-naming-static-constant-suffix), [`StaticConstantIgnoredRegexp`](#readability-identifier-naming-static-constant-ignored-regexp),
  [`StaticConstantHungarianPrefix`](#readability-identifier-naming-static-constant-hungarian-prefix)
- [`StaticVariableCase`](#readability-identifier-naming-static-variable-case), [`StaticVariablePrefix`](#readability-identifier-naming-static-variable-prefix),
  [`StaticVariableSuffix`](#readability-identifier-naming-static-variable-suffix), [`StaticVariableIgnoredRegexp`](#readability-identifier-naming-static-variable-ignored-regexp),
  [`StaticVariableHungarianPrefix`](#readability-identifier-naming-static-variable-hungarian-prefix)
- [`StructCase`](#readability-identifier-naming-struct-case), [`StructPrefix`](#readability-identifier-naming-struct-prefix), [`StructSuffix`](#readability-identifier-naming-struct-suffix),
  [`StructIgnoredRegexp`](#readability-identifier-naming-struct-ignored-regexp)
- [`TemplateParameterCase`](#readability-identifier-naming-template-parameter-case), [`TemplateParameterPrefix`](#readability-identifier-naming-template-parameter-prefix),
  [`TemplateParameterSuffix`](#readability-identifier-naming-template-parameter-suffix), [`TemplateParameterIgnoredRegexp`](#readability-identifier-naming-template-parameter-ignored-regexp)
- [`TemplateTemplateParameterCase`](#readability-identifier-naming-template-template-parameter-case),
  [`TemplateTemplateParameterPrefix`](#readability-identifier-naming-template-template-parameter-prefix),
  [`TemplateTemplateParameterSuffix`](#readability-identifier-naming-template-template-parameter-suffix),
  [`TemplateTemplateParameterIgnoredRegexp`](#readability-identifier-naming-template-template-parameter-ignored-regexp)
- [`TypeAliasCase`](#readability-identifier-naming-type-alias-case), [`TypeAliasPrefix`](#readability-identifier-naming-type-alias-prefix),
  [`TypeAliasSuffix`](#readability-identifier-naming-type-alias-suffix), [`TypeAliasIgnoredRegexp`](#readability-identifier-naming-type-alias-ignored-regexp)
- [`TypedefCase`](#readability-identifier-naming-typedef-case), [`TypedefPrefix`](#readability-identifier-naming-typedef-prefix), [`TypedefSuffix`](#readability-identifier-naming-typedef-suffix),
  [`TypedefIgnoredRegexp`](#readability-identifier-naming-typedef-ignored-regexp)
- [`TypeTemplateParameterCase`](#readability-identifier-naming-type-template-parameter-case),
  [`TypeTemplateParameterPrefix`](#readability-identifier-naming-type-template-parameter-prefix),
  [`TypeTemplateParameterSuffix`](#readability-identifier-naming-type-template-parameter-suffix),
  [`TypeTemplateParameterIgnoredRegexp`](#readability-identifier-naming-type-template-parameter-ignored-regexp)
- [`UnionCase`](#readability-identifier-naming-union-case), [`UnionPrefix`](#readability-identifier-naming-union-prefix), [`UnionSuffix`](#readability-identifier-naming-union-suffix),
  [`UnionIgnoredRegexp`](#readability-identifier-naming-union-ignored-regexp)
- [`ValueTemplateParameterCase`](#readability-identifier-naming-value-template-parameter-case),
  [`ValueTemplateParameterPrefix`](#readability-identifier-naming-value-template-parameter-prefix),
  [`ValueTemplateParameterSuffix`](#readability-identifier-naming-value-template-parameter-suffix),
  [`ValueTemplateParameterIgnoredRegexp`](#readability-identifier-naming-value-template-parameter-ignored-regexp)
- [`VariableCase`](#readability-identifier-naming-variable-case), [`VariablePrefix`](#readability-identifier-naming-variable-prefix), [`VariableSuffix`](#readability-identifier-naming-variable-suffix),
  [`VariableIgnoredRegexp`](#readability-identifier-naming-variable-ignored-regexp), [`VariableHungarianPrefix`](#readability-identifier-naming-variable-hungarian-prefix)
- [`VirtualMethodCase`](#readability-identifier-naming-virtual-method-case), [`VirtualMethodPrefix`](#readability-identifier-naming-virtual-method-prefix),
  [`VirtualMethodSuffix`](#readability-identifier-naming-virtual-method-suffix), [`VirtualMethodIgnoredRegexp`](#readability-identifier-naming-virtual-method-ignored-regexp)

## Options description

A detailed description of each option is presented below:

(readability-identifier-naming-default-case)=

```{option} DefaultCase
When defined, the check will ensure all names by default conform to the
selected casing.
```

(readability-identifier-naming-default-prefix)=

```{option} DefaultPrefix
When defined, the check will ensure all names by default will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-default-ignored-regexp)=

```{option} DefaultIgnoredRegexp
Identifier naming checks won't be enforced for all names by default
matching this regular expression.
```

(readability-identifier-naming-default-suffix)=

```{option} DefaultSuffix
When defined, the check will ensure all names by default will add the
suffix with the given value (regardless of casing).
```

(readability-identifier-naming-default-hungarian-prefix)=

```{option} DefaultHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

(readability-identifier-naming-abstract-class-case)=

```{option} AbstractClassCase
When defined, the check will ensure abstract class names conform to the
selected casing.
```

(readability-identifier-naming-abstract-class-prefix)=

```{option} AbstractClassPrefix
When defined, the check will ensure abstract class names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-abstract-class-ignored-regexp)=

```{option} AbstractClassIgnoredRegexp
Identifier naming checks won't be enforced for abstract class names
matching this regular expression.
```

(readability-identifier-naming-abstract-class-suffix)=

```{option} AbstractClassSuffix
When defined, the check will ensure abstract class names will add the
suffix with the given value (regardless of casing).
```

(readability-identifier-naming-abstract-class-hungarian-prefix)=

```{option} AbstractClassHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- [`AbstractClassCase`](#readability-identifier-naming-abstract-class-case) of `lower_case`
- [`AbstractClassPrefix`](#readability-identifier-naming-abstract-class-prefix) of `pre_`
- [`AbstractClassSuffix`](#readability-identifier-naming-abstract-class-suffix) of `_post`
- [`AbstractClassHungarianPrefix`](#readability-identifier-naming-abstract-class-hungarian-prefix) of `On`

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

(readability-identifier-naming-aggressive-dependent-member-lookup)=

```{option} AggressiveDependentMemberLookup
When `true`, the check will look in dependent base classes for dependent
member references that need changing. This can lead to errors with template
specializations. Default is `false`.
```

For example using values of:

- [`ClassMemberCase`](#readability-identifier-naming-class-member-case) of `lower_case`

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

After if [`AggressiveDependentMemberLookup`](#readability-identifier-naming-aggressive-dependent-member-lookup) is `false`:

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

After if [`AggressiveDependentMemberLookup`](#readability-identifier-naming-aggressive-dependent-member-lookup) is `true`:

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

(readability-identifier-naming-allow-trailing-underscore)=

```{option} AllowTrailingUnderscore
When `true`, a single trailing underscore is allowed on any identifier, in
addition to whatever casing, prefix and suffix are otherwise configured for
its kind.
```

For example using values:

- [`AllowTrailingUnderscore`](#readability-identifier-naming-allow-trailing-underscore)
  is `true`
- [`LocalVariableCase`](#readability-identifier-naming-local-variable-case) is
  `camelBack`

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

(readability-identifier-naming-check-anon-field-in-parent)=

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

If [`CheckAnonFieldInParent`](#readability-identifier-naming-check-anon-field-in-parent) is `false`, you may get warnings
that `iv_` and `fv_` are not coherent to public member names, because
`iv_` and `fv_` are public members of the anonymous union. When
[`CheckAnonFieldInParent`](#readability-identifier-naming-check-anon-field-in-parent) is `true`, `iv_` and `fv_` will be
treated as private data members of `Foo` for the purpose of name checking
and thus no warnings will be emitted.

(readability-identifier-naming-class-case)=

```{option} ClassCase
When defined, the check will ensure class names conform to the
selected casing.
```

(readability-identifier-naming-class-prefix)=

```{option} ClassPrefix
When defined, the check will ensure class names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-class-ignored-regexp)=

```{option} ClassIgnoredRegexp
Identifier naming checks won't be enforced for class names matching
this regular expression.
```

(readability-identifier-naming-class-suffix)=

```{option} ClassSuffix
When defined, the check will ensure class names will add the
suffix with the given value (regardless of casing).
```

(readability-identifier-naming-class-hungarian-prefix)=

```{option} ClassHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- [`ClassCase`](#readability-identifier-naming-class-case) of `lower_case`
- [`ClassPrefix`](#readability-identifier-naming-class-prefix) of `pre_`
- [`ClassSuffix`](#readability-identifier-naming-class-suffix) of `_post`
- [`ClassHungarianPrefix`](#readability-identifier-naming-class-hungarian-prefix) of `On`

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

(readability-identifier-naming-class-constexpr-case)=

```{option} ClassConstexprCase
When defined, the check will ensure class `constexpr` names conform to
the selected casing.
```

(readability-identifier-naming-class-constexpr-prefix)=

```{option} ClassConstexprPrefix
When defined, the check will ensure class `constexpr` names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-class-constexpr-ignored-regexp)=

```{option} ClassConstexprIgnoredRegexp
Identifier naming checks won't be enforced for class `constexpr` names
matching this regular expression.
```

(readability-identifier-naming-class-constexpr-suffix)=

```{option} ClassConstexprSuffix
When defined, the check will ensure class `constexpr` names will add the
suffix with the given value (regardless of casing).
```

(readability-identifier-naming-class-constexpr-hungarian-prefix)=

```{option} ClassConstexprHungarianPrefix
When enabled, the check ensures that the declared identifier will have a
Hungarian notation prefix based on the declared type.
```

For example using values of:

- [`ClassConstexprCase`](#readability-identifier-naming-class-constexpr-case) of `lower_case`
- [`ClassConstexprPrefix`](#readability-identifier-naming-class-constexpr-prefix) of `pre_`
- [`ClassConstexprSuffix`](#readability-identifier-naming-class-constexpr-suffix) of `_post`
- [`ClassConstexprHungarianPrefix`](#readability-identifier-naming-class-constexpr-hungarian-prefix) of `On`

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

(readability-identifier-naming-class-constant-case)=

```{option} ClassConstantCase
When defined, the check will ensure class constant names conform to the
selected casing.
```

(readability-identifier-naming-class-constant-prefix)=

```{option} ClassConstantPrefix
When defined, the check will ensure class constant names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-class-constant-ignored-regexp)=

```{option} ClassConstantIgnoredRegexp
Identifier naming checks won't be enforced for class constant names
matching this regular expression.
```

(readability-identifier-naming-class-constant-suffix)=

```{option} ClassConstantSuffix
When defined, the check will ensure class constant names will add the
suffix with the given value (regardless of casing).
```

(readability-identifier-naming-class-constant-hungarian-prefix)=

```{option} ClassConstantHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- [`ClassConstantCase`](#readability-identifier-naming-class-constant-case) of `lower_case`
- [`ClassConstantPrefix`](#readability-identifier-naming-class-constant-prefix) of `pre_`
- [`ClassConstantSuffix`](#readability-identifier-naming-class-constant-suffix) of `_post`
- [`ClassConstantHungarianPrefix`](#readability-identifier-naming-class-constant-hungarian-prefix) of `On`

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

(readability-identifier-naming-class-member-case)=

```{option} ClassMemberCase
When defined, the check will ensure class member names conform to the
selected casing.
```

(readability-identifier-naming-class-member-prefix)=

```{option} ClassMemberPrefix
When defined, the check will ensure class member names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-class-member-ignored-regexp)=

```{option} ClassMemberIgnoredRegexp
Identifier naming checks won't be enforced for class member names
matching this regular expression.
```

(readability-identifier-naming-class-member-suffix)=

```{option} ClassMemberSuffix
When defined, the check will ensure class member names will add the
suffix with the given value (regardless of casing).
```

(readability-identifier-naming-class-member-hungarian-prefix)=

```{option} ClassMemberHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- [`ClassMemberCase`](#readability-identifier-naming-class-member-case) of `lower_case`
- [`ClassMemberPrefix`](#readability-identifier-naming-class-member-prefix) of `pre_`
- [`ClassMemberSuffix`](#readability-identifier-naming-class-member-suffix) of `_post`
- [`ClassMemberHungarianPrefix`](#readability-identifier-naming-class-member-hungarian-prefix) of `On`

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

(readability-identifier-naming-class-method-case)=

```{option} ClassMethodCase
When defined, the check will ensure class method names conform to the
selected casing.
```

(readability-identifier-naming-class-method-prefix)=

```{option} ClassMethodPrefix
When defined, the check will ensure class method names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-class-method-ignored-regexp)=

```{option} ClassMethodIgnoredRegexp
Identifier naming checks won't be enforced for class method names
matching this regular expression.
```

(readability-identifier-naming-class-method-suffix)=

```{option} ClassMethodSuffix
When defined, the check will ensure class method names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- [`ClassMethodCase`](#readability-identifier-naming-class-method-case) of `lower_case`
- [`ClassMethodPrefix`](#readability-identifier-naming-class-method-prefix) of `pre_`
- [`ClassMethodSuffix`](#readability-identifier-naming-class-method-suffix) of `_post`

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

(readability-identifier-naming-concept-case)=

```{option} ConceptCase
When defined, the check will ensure concept names conform to the
selected casing.
```

(readability-identifier-naming-concept-prefix)=

```{option} ConceptPrefix
When defined, the check will ensure concept names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-concept-ignored-regexp)=

```{option} ConceptIgnoredRegexp
Identifier naming checks won't be enforced for concept names
matching this regular expression.
```

(readability-identifier-naming-concept-suffix)=

```{option} ConceptSuffix
When defined, the check will ensure concept names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- [`ConceptCase`](#readability-identifier-naming-concept-case) of `CamelCase`
- [`ConceptPrefix`](#readability-identifier-naming-concept-prefix) of `Pre`
- [`ConceptSuffix`](#readability-identifier-naming-concept-suffix) of `Post`

Identifies and/or transforms concept names as follows:

Before:

```c++
template<typename T> concept my_concept = requires (T t) { {t++}; };
```

After:

```c++
template<typename T> concept PreMyConceptPost = requires (T t) { {t++}; };
```

(readability-identifier-naming-constant-case)=

```{option} ConstantCase
When defined, the check will ensure constant names conform to the
selected casing.
```

(readability-identifier-naming-constant-prefix)=

```{option} ConstantPrefix
When defined, the check will ensure constant names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-constant-ignored-regexp)=

```{option} ConstantIgnoredRegexp
Identifier naming checks won't be enforced for constant names
matching this regular expression.
```

(readability-identifier-naming-constant-suffix)=

```{option} ConstantSuffix
When defined, the check will ensure constant names will add the
suffix with the given value (regardless of casing).
```

(readability-identifier-naming-constant-hungarian-prefix)=

```{option} ConstantHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- [`ConstantCase`](#readability-identifier-naming-constant-case) of `lower_case`
- [`ConstantPrefix`](#readability-identifier-naming-constant-prefix) of `pre_`
- [`ConstantSuffix`](#readability-identifier-naming-constant-suffix) of `_post`
- [`ConstantHungarianPrefix`](#readability-identifier-naming-constant-hungarian-prefix) of `On`

Identifies and/or transforms constant names as follows:

Before:

```c++
void function() { unsigned const MyConst_array[] = {1, 2, 3}; }
```

After:

```c++
void function() { unsigned const pre_myconst_array_post[] = {1, 2, 3}; }
```

(readability-identifier-naming-constant-member-case)=

```{option} ConstantMemberCase
When defined, the check will ensure constant member names conform to the
selected casing.
```

(readability-identifier-naming-constant-member-prefix)=

```{option} ConstantMemberPrefix
When defined, the check will ensure constant member names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-constant-member-ignored-regexp)=

```{option} ConstantMemberIgnoredRegexp
Identifier naming checks won't be enforced for constant member names
matching this regular expression.
```

(readability-identifier-naming-constant-member-suffix)=

```{option} ConstantMemberSuffix
When defined, the check will ensure constant member names will add the
suffix with the given value (regardless of casing).
```

(readability-identifier-naming-constant-member-hungarian-prefix)=

```{option} ConstantMemberHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- [`ConstantMemberCase`](#readability-identifier-naming-constant-member-case) of `lower_case`
- [`ConstantMemberPrefix`](#readability-identifier-naming-constant-member-prefix) of `pre_`
- [`ConstantMemberSuffix`](#readability-identifier-naming-constant-member-suffix) of `_post`
- [`ConstantMemberHungarianPrefix`](#readability-identifier-naming-constant-member-hungarian-prefix) of `On`

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

(readability-identifier-naming-constant-parameter-case)=

```{option} ConstantParameterCase
When defined, the check will ensure constant parameter names conform to the
selected casing.
```

(readability-identifier-naming-constant-parameter-prefix)=

```{option} ConstantParameterPrefix
When defined, the check will ensure constant parameter names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-constant-parameter-ignored-regexp)=

```{option} ConstantParameterIgnoredRegexp
Identifier naming checks won't be enforced for constant parameter names
matching this regular expression.
```

(readability-identifier-naming-constant-parameter-suffix)=

```{option} ConstantParameterSuffix
When defined, the check will ensure constant parameter names will add the
suffix with the given value (regardless of casing).
```

(readability-identifier-naming-constant-parameter-hungarian-prefix)=

```{option} ConstantParameterHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- [`ConstantParameterCase`](#readability-identifier-naming-constant-parameter-case) of `lower_case`
- [`ConstantParameterPrefix`](#readability-identifier-naming-constant-parameter-prefix) of `pre_`
- [`ConstantParameterSuffix`](#readability-identifier-naming-constant-parameter-suffix) of `_post`
- [`ConstantParameterHungarianPrefix`](#readability-identifier-naming-constant-parameter-hungarian-prefix) of `On`

Identifies and/or transforms constant parameter names as follows:

Before:

```c++
void GLOBAL_FUNCTION(int PARAMETER_1, int const CONST_parameter);
```

After:

```c++
void GLOBAL_FUNCTION(int PARAMETER_1, int const pre_const_parameter_post);
```

(readability-identifier-naming-constant-pointer-parameter-case)=

```{option} ConstantPointerParameterCase
When defined, the check will ensure constant pointer parameter names conform to the
selected casing.
```

(readability-identifier-naming-constant-pointer-parameter-prefix)=

```{option} ConstantPointerParameterPrefix
When defined, the check will ensure constant pointer parameter names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-constant-pointer-parameter-ignored-regexp)=

```{option} ConstantPointerParameterIgnoredRegexp
Identifier naming checks won't be enforced for constant pointer parameter
names matching this regular expression.
```

(readability-identifier-naming-constant-pointer-parameter-suffix)=

```{option} ConstantPointerParameterSuffix
When defined, the check will ensure constant pointer parameter names will add the
suffix with the given value (regardless of casing).
```

(readability-identifier-naming-constant-pointer-parameter-hungarian-prefix)=

```{option} ConstantPointerParameterHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- [`ConstantPointerParameterCase`](#readability-identifier-naming-constant-pointer-parameter-case) of `lower_case`
- [`ConstantPointerParameterPrefix`](#readability-identifier-naming-constant-pointer-parameter-prefix) of `pre_`
- [`ConstantPointerParameterSuffix`](#readability-identifier-naming-constant-pointer-parameter-suffix) of `_post`
- [`ConstantPointerParameterHungarianPrefix`](#readability-identifier-naming-constant-pointer-parameter-hungarian-prefix) of `On`

Identifies and/or transforms constant pointer parameter names as follows:

Before:

```c++
void GLOBAL_FUNCTION(int const *CONST_parameter);
```

After:

```c++
void GLOBAL_FUNCTION(int const *pre_const_parameter_post);
```

(readability-identifier-naming-constexpr-function-case)=

```{option} ConstexprFunctionCase
When defined, the check will ensure constexpr function names conform to the
selected casing.
```

(readability-identifier-naming-constexpr-function-prefix)=

```{option} ConstexprFunctionPrefix
When defined, the check will ensure constexpr function names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-constexpr-function-ignored-regexp)=

```{option} ConstexprFunctionIgnoredRegexp
Identifier naming checks won't be enforced for constexpr function names
matching this regular expression.
```

(readability-identifier-naming-constexpr-function-suffix)=

```{option} ConstexprFunctionSuffix
When defined, the check will ensure constexpr function names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- [`ConstexprFunctionCase`](#readability-identifier-naming-constexpr-function-case) of `lower_case`
- [`ConstexprFunctionPrefix`](#readability-identifier-naming-constexpr-function-prefix) of `pre_`
- [`ConstexprFunctionSuffix`](#readability-identifier-naming-constexpr-function-suffix) of `_post`

Identifies and/or transforms constexpr function names as follows:

Before:

```c++
constexpr int CE_function() { return 3; }
```

After:

```c++
constexpr int pre_ce_function_post() { return 3; }
```

(readability-identifier-naming-constexpr-method-case)=

```{option} ConstexprMethodCase
When defined, the check will ensure constexpr method names conform to the
selected casing.
```

(readability-identifier-naming-constexpr-method-prefix)=

```{option} ConstexprMethodPrefix
When defined, the check will ensure constexpr method names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-constexpr-method-ignored-regexp)=

```{option} ConstexprMethodIgnoredRegexp
Identifier naming checks won't be enforced for constexpr method names
matching this regular expression.
```

(readability-identifier-naming-constexpr-method-suffix)=

```{option} ConstexprMethodSuffix
When defined, the check will ensure constexpr method names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- [`ConstexprMethodCase`](#readability-identifier-naming-constexpr-method-case) of `lower_case`
- [`ConstexprMethodPrefix`](#readability-identifier-naming-constexpr-method-prefix) of `pre_`
- [`ConstexprMethodSuffix`](#readability-identifier-naming-constexpr-method-suffix) of `_post`

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

(readability-identifier-naming-constexpr-variable-case)=

```{option} ConstexprVariableCase
When defined, the check will ensure constexpr variable names conform to the
selected casing.
```

(readability-identifier-naming-constexpr-variable-prefix)=

```{option} ConstexprVariablePrefix
When defined, the check will ensure constexpr variable names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-constexpr-variable-ignored-regexp)=

```{option} ConstexprVariableIgnoredRegexp
Identifier naming checks won't be enforced for constexpr variable names
matching this regular expression.
```

(readability-identifier-naming-constexpr-variable-suffix)=

```{option} ConstexprVariableSuffix
When defined, the check will ensure constexpr variable names will add the
suffix with the given value (regardless of casing).
```

(readability-identifier-naming-constexpr-variable-hungarian-prefix)=

```{option} ConstexprVariableHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- [`ConstexprVariableCase`](#readability-identifier-naming-constexpr-variable-case) of `lower_case`
- [`ConstexprVariablePrefix`](#readability-identifier-naming-constexpr-variable-prefix) of `pre_`
- [`ConstexprVariableSuffix`](#readability-identifier-naming-constexpr-variable-suffix) of `_post`
- [`ConstexprVariableHungarianPrefix`](#readability-identifier-naming-constexpr-variable-hungarian-prefix) of `On`

Identifies and/or transforms constexpr variable names as follows:

Before:

```c++
constexpr int ConstExpr_variable = MyConstant;
```

After:

```c++
constexpr int pre_constexpr_variable_post = MyConstant;
```

(readability-identifier-naming-enum-case)=

```{option} EnumCase
When defined, the check will ensure enumeration names conform to the
selected casing.
```

(readability-identifier-naming-enum-prefix)=

```{option} EnumPrefix
When defined, the check will ensure enumeration names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-enum-ignored-regexp)=

```{option} EnumIgnoredRegexp
Identifier naming checks won't be enforced for enumeration names
matching this regular expression.
```

(readability-identifier-naming-enum-suffix)=

```{option} EnumSuffix
When defined, the check will ensure enumeration names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- [`EnumCase`](#readability-identifier-naming-enum-case) of `lower_case`
- [`EnumPrefix`](#readability-identifier-naming-enum-prefix) of `pre_`
- [`EnumSuffix`](#readability-identifier-naming-enum-suffix) of `_post`

Identifies and/or transforms enumeration names as follows:

Before:

```c++
enum FOO { One, Two, Three };
```

After:

```c++
enum pre_foo_post { One, Two, Three };
```

(readability-identifier-naming-enum-constant-case)=

```{option} EnumConstantCase
When defined, the check will ensure enumeration constant names conform to the
selected casing.
```

(readability-identifier-naming-enum-constant-prefix)=

```{option} EnumConstantPrefix
When defined, the check will ensure enumeration constant names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-enum-constant-ignored-regexp)=

```{option} EnumConstantIgnoredRegexp
Identifier naming checks won't be enforced for enumeration constant names
matching this regular expression.
```

(readability-identifier-naming-enum-constant-suffix)=

```{option} EnumConstantSuffix
When defined, the check will ensure enumeration constant names will add the
suffix with the given value (regardless of casing).
```

(readability-identifier-naming-enum-constant-hungarian-prefix)=

```{option} EnumConstantHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- [`EnumConstantCase`](#readability-identifier-naming-enum-constant-case) of `lower_case`
- [`EnumConstantPrefix`](#readability-identifier-naming-enum-constant-prefix) of `pre_`
- [`EnumConstantSuffix`](#readability-identifier-naming-enum-constant-suffix) of `_post`
- [`EnumConstantHungarianPrefix`](#readability-identifier-naming-enum-constant-hungarian-prefix) of `On`

Identifies and/or transforms enumeration constant names as follows:

Before:

```c++
enum FOO { One, Two, Three };
```

After:

```c++
enum FOO { pre_One_post, pre_Two_post, pre_Three_post };
```

(readability-identifier-naming-function-case)=

```{option} FunctionCase
When defined, the check will ensure function names conform to the
selected casing.
```

(readability-identifier-naming-function-prefix)=

```{option} FunctionPrefix
When defined, the check will ensure function names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-function-ignored-regexp)=

```{option} FunctionIgnoredRegexp
Identifier naming checks won't be enforced for function names
matching this regular expression.
```

(readability-identifier-naming-function-suffix)=

```{option} FunctionSuffix
When defined, the check will ensure function names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- [`FunctionCase`](#readability-identifier-naming-function-case) of `lower_case`
- [`FunctionPrefix`](#readability-identifier-naming-function-prefix) of `pre_`
- [`FunctionSuffix`](#readability-identifier-naming-function-suffix) of `_post`

Identifies and/or transforms function names as follows:

Before:

```c++
char MY_Function_string();
```

After:

```c++
char pre_my_function_string_post();
```

(readability-identifier-naming-get-config-per-file)=

```{option} GetConfigPerFile
When `true`, the check will look for the configuration for where an
identifier is declared. Useful for when included header files use a
different style.
Default is `true`.
```

(readability-identifier-naming-global-constexpr-variable-case)=

```{option} GlobalConstexprVariableCase
When defined, the check will ensure global `constexpr` variable names
conform to the selected casing.
```

(readability-identifier-naming-global-constexpr-variable-prefix)=

```{option} GlobalConstexprVariablePrefix
When defined, the check will ensure global `constexpr` variable names
will add the prefixed with the given value (regardless of casing).
```

(readability-identifier-naming-global-constexpr-variable-ignored-regexp)=

```{option} GlobalConstexprVariableIgnoredRegexp
Identifier naming checks won't be enforced for global `constexpr`
variable names matching this regular expression.
```

(readability-identifier-naming-global-constexpr-variable-suffix)=

```{option} GlobalConstexprVariableSuffix
When defined, the check will ensure global `constexpr` variable names
will add the suffix with the given value (regardless of casing).
```

(readability-identifier-naming-global-constexpr-variable-hungarian-prefix)=

```{option} GlobalConstexprVariableHungarianPrefix
When enabled, the check ensures that the declared identifier will have a
Hungarian notation prefix based on the declared type.
```

For example using values of:

- [`GlobalConstexprVariableCase`](#readability-identifier-naming-global-constexpr-variable-case) of `lower_case`
- [`GlobalConstexprVariablePrefix`](#readability-identifier-naming-global-constexpr-variable-prefix) of `pre_`
- [`GlobalConstexprVariableSuffix`](#readability-identifier-naming-global-constexpr-variable-suffix) of `_post`
- [`GlobalConstexprVariableHungarianPrefix`](#readability-identifier-naming-global-constexpr-variable-hungarian-prefix) of `On`

Identifies and/or transforms global `constexpr` variable names as follows:

Before:

```c++
constexpr unsigned ImportantValue = 69;
```

After:

```c++
constexpr unsigned pre_important_value_post = 69;
```

(readability-identifier-naming-global-constant-case)=

```{option} GlobalConstantCase
When defined, the check will ensure global constant names conform to the
selected casing.
```

(readability-identifier-naming-global-constant-prefix)=

```{option} GlobalConstantPrefix
When defined, the check will ensure global constant names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-global-constant-ignored-regexp)=

```{option} GlobalConstantIgnoredRegexp
Identifier naming checks won't be enforced for global constant names
matching this regular expression.
```

(readability-identifier-naming-global-constant-suffix)=

```{option} GlobalConstantSuffix
When defined, the check will ensure global constant names will add the
suffix with the given value (regardless of casing).
```

(readability-identifier-naming-global-constant-hungarian-prefix)=

```{option} GlobalConstantHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- [`GlobalConstantCase`](#readability-identifier-naming-global-constant-case) of `lower_case`
- [`GlobalConstantPrefix`](#readability-identifier-naming-global-constant-prefix) of `pre_`
- [`GlobalConstantSuffix`](#readability-identifier-naming-global-constant-suffix) of `_post`
- [`GlobalConstantHungarianPrefix`](#readability-identifier-naming-global-constant-hungarian-prefix) of `On`

Identifies and/or transforms global constant names as follows:

Before:

```c++
unsigned const MyConstGlobal_array[] = {1, 2, 3};
```

After:

```c++
unsigned const pre_myconstglobal_array_post[] = {1, 2, 3};
```

(readability-identifier-naming-global-constant-pointer-case)=

```{option} GlobalConstantPointerCase
When defined, the check will ensure global constant pointer names conform to the
selected casing.
```

(readability-identifier-naming-global-constant-pointer-prefix)=

```{option} GlobalConstantPointerPrefix
When defined, the check will ensure global constant pointer names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-global-constant-pointer-ignored-regexp)=

```{option} GlobalConstantPointerIgnoredRegexp
Identifier naming checks won't be enforced for global constant pointer
names matching this regular expression.
```

(readability-identifier-naming-global-constant-pointer-suffix)=

```{option} GlobalConstantPointerSuffix
When defined, the check will ensure global constant pointer names will add the
suffix with the given value (regardless of casing).
```

(readability-identifier-naming-global-constant-pointer-hungarian-prefix)=

```{option} GlobalConstantPointerHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- [`GlobalConstantPointerCase`](#readability-identifier-naming-global-constant-pointer-case) of `lower_case`
- [`GlobalConstantPointerPrefix`](#readability-identifier-naming-global-constant-pointer-prefix) of `pre_`
- [`GlobalConstantPointerSuffix`](#readability-identifier-naming-global-constant-pointer-suffix) of `_post`
- [`GlobalConstantPointerHungarianPrefix`](#readability-identifier-naming-global-constant-pointer-hungarian-prefix) of `On`

Identifies and/or transforms global constant pointer names as follows:

Before:

```c++
int *const MyConstantGlobalPointer = nullptr;
```

After:

```c++
int *const pre_myconstantglobalpointer_post = nullptr;
```

(readability-identifier-naming-global-function-case)=

```{option} GlobalFunctionCase
When defined, the check will ensure global function names conform to the
selected casing.
```

(readability-identifier-naming-global-function-prefix)=

```{option} GlobalFunctionPrefix
When defined, the check will ensure global function names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-global-function-ignored-regexp)=

```{option} GlobalFunctionIgnoredRegexp
Identifier naming checks won't be enforced for global function names
matching this regular expression.
```

(readability-identifier-naming-global-function-suffix)=

```{option} GlobalFunctionSuffix
When defined, the check will ensure global function names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- [`GlobalFunctionCase`](#readability-identifier-naming-global-function-case) of `lower_case`
- [`GlobalFunctionPrefix`](#readability-identifier-naming-global-function-prefix) of `pre_`
- [`GlobalFunctionSuffix`](#readability-identifier-naming-global-function-suffix) of `_post`

Identifies and/or transforms global function names as follows:

Before:

```c++
void GLOBAL_FUNCTION(int PARAMETER_1, int const CONST_parameter);
```

After:

```c++
void pre_global_function_post(int PARAMETER_1, int const CONST_parameter);
```

(readability-identifier-naming-global-pointer-case)=

```{option} GlobalPointerCase
When defined, the check will ensure global pointer names conform to the
selected casing.
```

(readability-identifier-naming-global-pointer-prefix)=

```{option} GlobalPointerPrefix
When defined, the check will ensure global pointer names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-global-pointer-ignored-regexp)=

```{option} GlobalPointerIgnoredRegexp
Identifier naming checks won't be enforced for global pointer names
matching this regular expression.
```

(readability-identifier-naming-global-pointer-suffix)=

```{option} GlobalPointerSuffix
When defined, the check will ensure global pointer names will add the
suffix with the given value (regardless of casing).
```

(readability-identifier-naming-global-pointer-hungarian-prefix)=

```{option} GlobalPointerHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- [`GlobalPointerCase`](#readability-identifier-naming-global-pointer-case) of `lower_case`
- [`GlobalPointerPrefix`](#readability-identifier-naming-global-pointer-prefix) of `pre_`
- [`GlobalPointerSuffix`](#readability-identifier-naming-global-pointer-suffix) of `_post`
- [`GlobalPointerHungarianPrefix`](#readability-identifier-naming-global-pointer-hungarian-prefix) of `On`

Identifies and/or transforms global pointer names as follows:

Before:

```c++
int *GLOBAL3;
```

After:

```c++
int *pre_global3_post;
```

(readability-identifier-naming-global-variable-case)=

```{option} GlobalVariableCase
When defined, the check will ensure global variable names conform to the
selected casing.
```

(readability-identifier-naming-global-variable-prefix)=

```{option} GlobalVariablePrefix
When defined, the check will ensure global variable names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-global-variable-ignored-regexp)=

```{option} GlobalVariableIgnoredRegexp
Identifier naming checks won't be enforced for global variable names
matching this regular expression.
```

(readability-identifier-naming-global-variable-suffix)=

```{option} GlobalVariableSuffix
When defined, the check will ensure global variable names will add the
suffix with the given value (regardless of casing).
```

(readability-identifier-naming-global-variable-hungarian-prefix)=

```{option} GlobalVariableHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- [`GlobalVariableCase`](#readability-identifier-naming-global-variable-case) of `lower_case`
- [`GlobalVariablePrefix`](#readability-identifier-naming-global-variable-prefix) of `pre_`
- [`GlobalVariableSuffix`](#readability-identifier-naming-global-variable-suffix) of `_post`
- [`GlobalVariableHungarianPrefix`](#readability-identifier-naming-global-variable-hungarian-prefix) of `On`

Identifies and/or transforms global variable names as follows:

Before:

```c++
int GLOBAL3;
```

After:

```c++
int pre_global3_post;
```

(readability-identifier-naming-ignore-main-like-functions)=

```{option} IgnoreMainLikeFunctions
When `true`, functions that have a similar signature to `main` or
`wmain` won't enforce checks on the names of their parameters.
Default is `false`.
```

(readability-identifier-naming-inline-namespace-case)=

```{option} InlineNamespaceCase
When defined, the check will ensure inline namespaces names conform to the
selected casing.
```

(readability-identifier-naming-inline-namespace-prefix)=

```{option} InlineNamespacePrefix
When defined, the check will ensure inline namespaces names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-inline-namespace-ignored-regexp)=

```{option} InlineNamespaceIgnoredRegexp
Identifier naming checks won't be enforced for inline namespaces names
matching this regular expression.
```

(readability-identifier-naming-inline-namespace-suffix)=

```{option} InlineNamespaceSuffix
When defined, the check will ensure inline namespaces names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- [`InlineNamespaceCase`](#readability-identifier-naming-inline-namespace-case) of `lower_case`
- [`InlineNamespacePrefix`](#readability-identifier-naming-inline-namespace-prefix) of `pre_`
- [`InlineNamespaceSuffix`](#readability-identifier-naming-inline-namespace-suffix) of `_post`

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

(readability-identifier-naming-lambda-capture-case)=

```{option} LambdaCaptureCase
When defined, the check will ensure lambda init-capture names (e.g.
`Captured` in `[Captured = Var]`) conform to the selected casing.
A simple, non-init capture (e.g. `[Var]` or `[&Var]`) refers to the
same declaration as `Var` itself, so it keeps following whichever
naming style applies to `Var`'s own declaration instead.
```

(readability-identifier-naming-lambda-capture-prefix)=

```{option} LambdaCapturePrefix
When defined, the check will ensure lambda init-capture names will add
the prefix with the given value (regardless of casing).
```

(readability-identifier-naming-lambda-capture-ignored-regexp)=

```{option} LambdaCaptureIgnoredRegexp
Identifier naming checks won't be enforced for lambda init-capture names
matching this regular expression.
```

(readability-identifier-naming-lambda-capture-suffix)=

```{option} LambdaCaptureSuffix
When defined, the check will ensure lambda init-capture names will add
the suffix with the given value (regardless of casing).
```

(readability-identifier-naming-lambda-capture-hungarian-prefix)=

```{option} LambdaCaptureHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- [`LambdaCaptureCase`](#readability-identifier-naming-lambda-capture-case) of `CamelCase`
- [`LambdaCapturePrefix`](#readability-identifier-naming-lambda-capture-prefix) of `c_`

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

(readability-identifier-naming-local-constexpr-variable-case)=

```{option} LocalConstexprVariableCase
When defined, the check will ensure local `constexpr` variable names
conform to the selected casing.
```

(readability-identifier-naming-local-constexpr-variable-prefix)=

```{option} LocalConstexprVariablePrefix
When defined, the check will ensure local `constexpr` variable names will
add the prefixed with the given value (regardless of casing).
```

(readability-identifier-naming-local-constexpr-variable-ignored-regexp)=

```{option} LocalConstexprVariableIgnoredRegexp
Identifier naming checks won't be enforced for local `constexpr` variable
names matching this regular expression.
```

(readability-identifier-naming-local-constexpr-variable-suffix)=

```{option} LocalConstexprVariableSuffix
When defined, the check will ensure local `constexpr` variable names will
add the suffix with the given value (regardless of casing).
```

(readability-identifier-naming-local-constexpr-variable-hungarian-prefix)=

```{option} LocalConstexprVariableHungarianPrefix
When enabled, the check ensures that the declared identifier will have a
Hungarian notation prefix based on the declared type.
```

For example using values of:

- [`LocalConstexprVariableCase`](#readability-identifier-naming-local-constexpr-variable-case) of `lower_case`
- [`LocalConstexprVariablePrefix`](#readability-identifier-naming-local-constexpr-variable-prefix) of `pre_`
- [`LocalConstexprVariableSuffix`](#readability-identifier-naming-local-constexpr-variable-suffix) of `_post`
- [`LocalConstexprVariableHungarianPrefix`](#readability-identifier-naming-local-constexpr-variable-hungarian-prefix) of `On`

Identifies and/or transforms local `constexpr` variable names as follows:

Before:

```c++
void foo() { int const local_Constexpr = 420; }
```

After:

```c++
void foo() { int const pre_local_constexpr_post = 420; }
```

(readability-identifier-naming-local-constant-case)=

```{option} LocalConstantCase
When defined, the check will ensure local constant names conform to the
selected casing.
```

(readability-identifier-naming-local-constant-prefix)=

```{option} LocalConstantPrefix
When defined, the check will ensure local constant names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-local-constant-ignored-regexp)=

```{option} LocalConstantIgnoredRegexp
Identifier naming checks won't be enforced for local constant names
matching this regular expression.
```

(readability-identifier-naming-local-constant-suffix)=

```{option} LocalConstantSuffix
When defined, the check will ensure local constant names will add the
suffix with the given value (regardless of casing).
```

(readability-identifier-naming-local-constant-hungarian-prefix)=

```{option} LocalConstantHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- [`LocalConstantCase`](#readability-identifier-naming-local-constant-case) of `lower_case`
- [`LocalConstantPrefix`](#readability-identifier-naming-local-constant-prefix) of `pre_`
- [`LocalConstantSuffix`](#readability-identifier-naming-local-constant-suffix) of `_post`
- [`LocalConstantHungarianPrefix`](#readability-identifier-naming-local-constant-hungarian-prefix) of `On`

Identifies and/or transforms local constant names as follows:

Before:

```c++
void foo() { int const local_Constant = 3; }
```

After:

```c++
void foo() { int const pre_local_constant_post = 3; }
```

(readability-identifier-naming-local-constant-pointer-case)=

```{option} LocalConstantPointerCase
When defined, the check will ensure local constant pointer names conform to the
selected casing.
```

(readability-identifier-naming-local-constant-pointer-prefix)=

```{option} LocalConstantPointerPrefix
When defined, the check will ensure local constant pointer names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-local-constant-pointer-ignored-regexp)=

```{option} LocalConstantPointerIgnoredRegexp
Identifier naming checks won't be enforced for local constant pointer names
matching this regular expression.
```

(readability-identifier-naming-local-constant-pointer-suffix)=

```{option} LocalConstantPointerSuffix
When defined, the check will ensure local constant pointer names will add the
suffix with the given value (regardless of casing).
```

(readability-identifier-naming-local-constant-pointer-hungarian-prefix)=

```{option} LocalConstantPointerHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- [`LocalConstantPointerCase`](#readability-identifier-naming-local-constant-pointer-case) of `lower_case`
- [`LocalConstantPointerPrefix`](#readability-identifier-naming-local-constant-pointer-prefix) of `pre_`
- [`LocalConstantPointerSuffix`](#readability-identifier-naming-local-constant-pointer-suffix) of `_post`
- [`LocalConstantPointerHungarianPrefix`](#readability-identifier-naming-local-constant-pointer-hungarian-prefix) of `On`

Identifies and/or transforms local constant pointer names as follows:

Before:

```c++
void foo() { int const *local_Constant = 3; }
```

After:

```c++
void foo() { int const *pre_local_constant_post = 3; }
```

(readability-identifier-naming-local-pointer-case)=

```{option} LocalPointerCase
When defined, the check will ensure local pointer names conform to the
selected casing.
```

(readability-identifier-naming-local-pointer-prefix)=

```{option} LocalPointerPrefix
When defined, the check will ensure local pointer names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-local-pointer-ignored-regexp)=

```{option} LocalPointerIgnoredRegexp
Identifier naming checks won't be enforced for local pointer names
matching this regular expression.
```

(readability-identifier-naming-local-pointer-suffix)=

```{option} LocalPointerSuffix
When defined, the check will ensure local pointer names will add the
suffix with the given value (regardless of casing).
```

(readability-identifier-naming-local-pointer-hungarian-prefix)=

```{option} LocalPointerHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- [`LocalPointerCase`](#readability-identifier-naming-local-pointer-case) of `lower_case`
- [`LocalPointerPrefix`](#readability-identifier-naming-local-pointer-prefix) of `pre_`
- [`LocalPointerSuffix`](#readability-identifier-naming-local-pointer-suffix) of `_post`
- [`LocalPointerHungarianPrefix`](#readability-identifier-naming-local-pointer-hungarian-prefix) of `On`

Identifies and/or transforms local pointer names as follows:

Before:

```c++
void foo() { int *local_Constant; }
```

After:

```c++
void foo() { int *pre_local_constant_post; }
```

(readability-identifier-naming-local-variable-case)=

```{option} LocalVariableCase
When defined, the check will ensure local variable names conform to the
selected casing.
```

(readability-identifier-naming-local-variable-prefix)=

```{option} LocalVariablePrefix
When defined, the check will ensure local variable names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-local-variable-ignored-regexp)=

```{option} LocalVariableIgnoredRegexp
Identifier naming checks won't be enforced for local variable names
matching this regular expression.
```

For example using values of:

- [`LocalVariableCase`](#readability-identifier-naming-local-variable-case) of `CamelCase`
- [`LocalVariableIgnoredRegexp`](#readability-identifier-naming-local-variable-ignored-regexp) of `\w{1,2}`

Will exclude variables with a length less than or equal to 2 from the
camel case check applied to other variables.

(readability-identifier-naming-local-variable-suffix)=

```{option} LocalVariableSuffix
When defined, the check will ensure local variable names will add the
suffix with the given value (regardless of casing).
```

(readability-identifier-naming-local-variable-hungarian-prefix)=

```{option} LocalVariableHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- [`LocalVariableCase`](#readability-identifier-naming-local-variable-case) of `lower_case`
- [`LocalVariablePrefix`](#readability-identifier-naming-local-variable-prefix) of `pre_`
- [`LocalVariableSuffix`](#readability-identifier-naming-local-variable-suffix) of `_post`
- [`LocalVariableHungarianPrefix`](#readability-identifier-naming-local-variable-hungarian-prefix) of `On`

Identifies and/or transforms local variable names as follows:

Before:

```c++
void foo() { int local_Constant; }
```

After:

```c++
void foo() { int pre_local_constant_post; }
```

(readability-identifier-naming-macro-definition-case)=

```{option} MacroDefinitionCase
When defined, the check will ensure macro definitions conform to the
selected casing.
```

(readability-identifier-naming-macro-definition-prefix)=

```{option} MacroDefinitionPrefix
When defined, the check will ensure macro definitions will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-macro-definition-ignored-regexp)=

```{option} MacroDefinitionIgnoredRegexp
Identifier naming checks won't be enforced for macro definitions
matching this regular expression.
```

(readability-identifier-naming-macro-definition-suffix)=

```{option} MacroDefinitionSuffix
When defined, the check will ensure macro definitions will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- [`MacroDefinitionCase`](#readability-identifier-naming-macro-definition-case) of `lower_case`
- [`MacroDefinitionPrefix`](#readability-identifier-naming-macro-definition-prefix) of `pre_`
- [`MacroDefinitionSuffix`](#readability-identifier-naming-macro-definition-suffix) of `_post`

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

(readability-identifier-naming-member-case)=

```{option} MemberCase
When defined, the check will ensure member names conform to the
selected casing.
```

(readability-identifier-naming-member-prefix)=

```{option} MemberPrefix
When defined, the check will ensure member names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-member-ignored-regexp)=

```{option} MemberIgnoredRegexp
Identifier naming checks won't be enforced for member names
matching this regular expression.
```

(readability-identifier-naming-member-suffix)=

```{option} MemberSuffix
When defined, the check will ensure member names will add the
suffix with the given value (regardless of casing).
```

(readability-identifier-naming-member-hungarian-prefix)=

```{option} MemberHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- [`MemberCase`](#readability-identifier-naming-member-case) of `lower_case`
- [`MemberPrefix`](#readability-identifier-naming-member-prefix) of `pre_`
- [`MemberSuffix`](#readability-identifier-naming-member-suffix) of `_post`
- [`MemberHungarianPrefix`](#readability-identifier-naming-member-hungarian-prefix) of `On`

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

(readability-identifier-naming-method-case)=

```{option} MethodCase
When defined, the check will ensure method names conform to the
selected casing.
```

(readability-identifier-naming-method-prefix)=

```{option} MethodPrefix
When defined, the check will ensure method names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-method-ignored-regexp)=

```{option} MethodIgnoredRegexp
Identifier naming checks won't be enforced for method names
matching this regular expression.
```

(readability-identifier-naming-method-suffix)=

```{option} MethodSuffix
When defined, the check will ensure method names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- [`MethodCase`](#readability-identifier-naming-method-case) of `lower_case`
- [`MethodPrefix`](#readability-identifier-naming-method-prefix) of `pre_`
- [`MethodSuffix`](#readability-identifier-naming-method-suffix) of `_post`

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

(readability-identifier-naming-namespace-case)=

```{option} NamespaceCase
When defined, the check will ensure namespace names conform to the
selected casing.
```

(readability-identifier-naming-namespace-prefix)=

```{option} NamespacePrefix
When defined, the check will ensure namespace names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-namespace-ignored-regexp)=

```{option} NamespaceIgnoredRegexp
Identifier naming checks won't be enforced for namespace names
matching this regular expression.
```

(readability-identifier-naming-namespace-suffix)=

```{option} NamespaceSuffix
When defined, the check will ensure namespace names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- [`NamespaceCase`](#readability-identifier-naming-namespace-case) of `lower_case`
- [`NamespacePrefix`](#readability-identifier-naming-namespace-prefix) of `pre_`
- [`NamespaceSuffix`](#readability-identifier-naming-namespace-suffix) of `_post`

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

(readability-identifier-naming-parameter-case)=

```{option} ParameterCase
When defined, the check will ensure parameter names conform to the
selected casing.
```

(readability-identifier-naming-parameter-prefix)=

```{option} ParameterPrefix
When defined, the check will ensure parameter names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-parameter-ignored-regexp)=

```{option} ParameterIgnoredRegexp
Identifier naming checks won't be enforced for parameter names
matching this regular expression.
```

(readability-identifier-naming-parameter-suffix)=

```{option} ParameterSuffix
When defined, the check will ensure parameter names will add the
suffix with the given value (regardless of casing).
```

(readability-identifier-naming-parameter-hungarian-prefix)=

```{option} ParameterHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- [`ParameterCase`](#readability-identifier-naming-parameter-case) of `lower_case`
- [`ParameterPrefix`](#readability-identifier-naming-parameter-prefix) of `pre_`
- [`ParameterSuffix`](#readability-identifier-naming-parameter-suffix) of `_post`
- [`ParameterHungarianPrefix`](#readability-identifier-naming-parameter-hungarian-prefix) of `On`

Identifies and/or transforms parameter names as follows:

Before:

```c++
void GLOBAL_FUNCTION(int PARAMETER_1, int const CONST_parameter);
```

After:

```c++
void GLOBAL_FUNCTION(int pre_parameter_post, int const CONST_parameter);
```

(readability-identifier-naming-parameter-pack-case)=

```{option} ParameterPackCase
When defined, the check will ensure parameter pack names conform to the
selected casing.
```

(readability-identifier-naming-parameter-pack-prefix)=

```{option} ParameterPackPrefix
When defined, the check will ensure parameter pack names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-parameter-pack-ignored-regexp)=

```{option} ParameterPackIgnoredRegexp
Identifier naming checks won't be enforced for parameter pack names
matching this regular expression.
```

(readability-identifier-naming-parameter-pack-suffix)=

```{option} ParameterPackSuffix
When defined, the check will ensure parameter pack names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- [`ParameterPackCase`](#readability-identifier-naming-parameter-pack-case) of `lower_case`
- [`ParameterPackPrefix`](#readability-identifier-naming-parameter-pack-prefix) of `pre_`
- [`ParameterPackSuffix`](#readability-identifier-naming-parameter-pack-suffix) of `_post`

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

(readability-identifier-naming-pointer-parameter-case)=

```{option} PointerParameterCase
When defined, the check will ensure pointer parameter names conform to the
selected casing.
```

(readability-identifier-naming-pointer-parameter-prefix)=

```{option} PointerParameterPrefix
When defined, the check will ensure pointer parameter names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-pointer-parameter-ignored-regexp)=

```{option} PointerParameterIgnoredRegexp
Identifier naming checks won't be enforced for pointer parameter names
matching this regular expression.
```

(readability-identifier-naming-pointer-parameter-suffix)=

```{option} PointerParameterSuffix
When defined, the check will ensure pointer parameter names will add the
suffix with the given value (regardless of casing).
```

(readability-identifier-naming-pointer-parameter-hungarian-prefix)=

```{option} PointerParameterHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- [`PointerParameterCase`](#readability-identifier-naming-pointer-parameter-case) of `lower_case`
- [`PointerParameterPrefix`](#readability-identifier-naming-pointer-parameter-prefix) of `pre_`
- [`PointerParameterSuffix`](#readability-identifier-naming-pointer-parameter-suffix) of `_post`
- [`PointerParameterHungarianPrefix`](#readability-identifier-naming-pointer-parameter-hungarian-prefix) of `On`

Identifies and/or transforms pointer parameter names as follows:

Before:

```c++
void FUNCTION(int *PARAMETER);
```

After:

```c++
void FUNCTION(int *pre_parameter_post);
```

(readability-identifier-naming-private-member-case)=

```{option} PrivateMemberCase
When defined, the check will ensure private member names conform to the
selected casing.
```

(readability-identifier-naming-private-member-prefix)=

```{option} PrivateMemberPrefix
When defined, the check will ensure private member names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-private-member-ignored-regexp)=

```{option} PrivateMemberIgnoredRegexp
Identifier naming checks won't be enforced for private member names
matching this regular expression.
```

(readability-identifier-naming-private-member-suffix)=

```{option} PrivateMemberSuffix
When defined, the check will ensure private member names will add the
suffix with the given value (regardless of casing).
```

(readability-identifier-naming-private-member-hungarian-prefix)=

```{option} PrivateMemberHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- [`PrivateMemberCase`](#readability-identifier-naming-private-member-case) of `lower_case`
- [`PrivateMemberPrefix`](#readability-identifier-naming-private-member-prefix) of `pre_`
- [`PrivateMemberSuffix`](#readability-identifier-naming-private-member-suffix) of `_post`
- [`PrivateMemberHungarianPrefix`](#readability-identifier-naming-private-member-hungarian-prefix) of `On`

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

(readability-identifier-naming-private-method-case)=

```{option} PrivateMethodCase
When defined, the check will ensure private method names conform to the
selected casing.
```

(readability-identifier-naming-private-method-prefix)=

```{option} PrivateMethodPrefix
When defined, the check will ensure private method names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-private-method-ignored-regexp)=

```{option} PrivateMethodIgnoredRegexp
Identifier naming checks won't be enforced for private method names
matching this regular expression.
```

(readability-identifier-naming-private-method-suffix)=

```{option} PrivateMethodSuffix
When defined, the check will ensure private method names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- [`PrivateMethodCase`](#readability-identifier-naming-private-method-case) of `lower_case`
- [`PrivateMethodPrefix`](#readability-identifier-naming-private-method-prefix) of `pre_`
- [`PrivateMethodSuffix`](#readability-identifier-naming-private-method-suffix) of `_post`

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

(readability-identifier-naming-protected-member-case)=

```{option} ProtectedMemberCase
When defined, the check will ensure protected member names conform to the
selected casing.
```

(readability-identifier-naming-protected-member-prefix)=

```{option} ProtectedMemberPrefix
When defined, the check will ensure protected member names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-protected-member-ignored-regexp)=

```{option} ProtectedMemberIgnoredRegexp
Identifier naming checks won't be enforced for protected member names
matching this regular expression.
```

(readability-identifier-naming-protected-member-suffix)=

```{option} ProtectedMemberSuffix
When defined, the check will ensure protected member names will add the
suffix with the given value (regardless of casing).
```

(readability-identifier-naming-protected-member-hungarian-prefix)=

```{option} ProtectedMemberHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- [`ProtectedMemberCase`](#readability-identifier-naming-protected-member-case) of `lower_case`
- [`ProtectedMemberPrefix`](#readability-identifier-naming-protected-member-prefix) of `pre_`
- [`ProtectedMemberSuffix`](#readability-identifier-naming-protected-member-suffix) of `_post`
- [`ProtectedMemberHungarianPrefix`](#readability-identifier-naming-protected-member-hungarian-prefix) of `On`

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

(readability-identifier-naming-protected-method-case)=

```{option} ProtectedMethodCase
When defined, the check will ensure protected method names conform to the
selected casing.
```

(readability-identifier-naming-protected-method-prefix)=

```{option} ProtectedMethodPrefix
When defined, the check will ensure protected method names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-protected-method-ignored-regexp)=

```{option} ProtectedMethodIgnoredRegexp
Identifier naming checks won't be enforced for protected method names
matching this regular expression.
```

(readability-identifier-naming-protected-method-suffix)=

```{option} ProtectedMethodSuffix
When defined, the check will ensure protected method names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- [`ProtectedMethodCase`](#readability-identifier-naming-protected-method-case) of `lower_case`
- [`ProtectedMethodPrefix`](#readability-identifier-naming-protected-method-prefix) of `pre_`
- [`ProtectedMethodSuffix`](#readability-identifier-naming-protected-method-suffix) of `_post`

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

(readability-identifier-naming-public-member-case)=

```{option} PublicMemberCase
When defined, the check will ensure public member names conform to the
selected casing.
```

(readability-identifier-naming-public-member-prefix)=

```{option} PublicMemberPrefix
When defined, the check will ensure public member names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-public-member-ignored-regexp)=

```{option} PublicMemberIgnoredRegexp
Identifier naming checks won't be enforced for public member names
matching this regular expression.
```

(readability-identifier-naming-public-member-suffix)=

```{option} PublicMemberSuffix
When defined, the check will ensure public member names will add the
suffix with the given value (regardless of casing).
```

(readability-identifier-naming-public-member-hungarian-prefix)=

```{option} PublicMemberHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- [`PublicMemberCase`](#readability-identifier-naming-public-member-case) of `lower_case`
- [`PublicMemberPrefix`](#readability-identifier-naming-public-member-prefix) of `pre_`
- [`PublicMemberSuffix`](#readability-identifier-naming-public-member-suffix) of `_post`
- [`PublicMemberHungarianPrefix`](#readability-identifier-naming-public-member-hungarian-prefix) of `On`

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

(readability-identifier-naming-public-method-case)=

```{option} PublicMethodCase
When defined, the check will ensure public method names conform to the
selected casing.
```

(readability-identifier-naming-public-method-prefix)=

```{option} PublicMethodPrefix
When defined, the check will ensure public method names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-public-method-ignored-regexp)=

```{option} PublicMethodIgnoredRegexp
Identifier naming checks won't be enforced for public method names
matching this regular expression.
```

(readability-identifier-naming-public-method-suffix)=

```{option} PublicMethodSuffix
When defined, the check will ensure public method names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- [`PublicMethodCase`](#readability-identifier-naming-public-method-case) of `lower_case`
- [`PublicMethodPrefix`](#readability-identifier-naming-public-method-prefix) of `pre_`
- [`PublicMethodSuffix`](#readability-identifier-naming-public-method-suffix) of `_post`

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

(readability-identifier-naming-scoped-enum-constant-case)=

```{option} ScopedEnumConstantCase
When defined, the check will ensure scoped enum constant names conform to
the selected casing.
```

(readability-identifier-naming-scoped-enum-constant-prefix)=

```{option} ScopedEnumConstantPrefix
When defined, the check will ensure scoped enum constant names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-scoped-enum-constant-ignored-regexp)=

```{option} ScopedEnumConstantIgnoredRegexp
Identifier naming checks won't be enforced for scoped enum constant names
matching this regular expression.
```

(readability-identifier-naming-scoped-enum-constant-suffix)=

```{option} ScopedEnumConstantSuffix
When defined, the check will ensure scoped enum constant names will add the
suffix with the given value (regardless of casing).
```

(readability-identifier-naming-scoped-enum-constant-hungarian-prefix)=

```{option} ScopedEnumConstantHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- [`ScopedEnumConstantCase`](#readability-identifier-naming-scoped-enum-constant-case) of `lower_case`
- [`ScopedEnumConstantPrefix`](#readability-identifier-naming-scoped-enum-constant-prefix) of `pre_`
- [`ScopedEnumConstantSuffix`](#readability-identifier-naming-scoped-enum-constant-suffix) of `_post`
- [`ScopedEnumConstantHungarianPrefix`](#readability-identifier-naming-scoped-enum-constant-hungarian-prefix) of `On`

Identifies and/or transforms enumeration constant names as follows:

Before:

```c++
enum class FOO { One, Two, Three };
```

After:

```c++
enum class FOO { pre_One_post, pre_Two_post, pre_Three_post };
```

(readability-identifier-naming-static-constexpr-variable-case)=

```{option} StaticConstexprVariableCase
When defined, the check will ensure static `constexpr` variable names
conform to the selected casing.
```

(readability-identifier-naming-static-constexpr-variable-prefix)=

```{option} StaticConstexprVariablePrefix
When defined, the check will ensure static `constexpr` variable names
will add the prefixed with the given value (regardless of casing).
```

(readability-identifier-naming-static-constexpr-variable-ignored-regexp)=

```{option} StaticConstexprVariableIgnoredRegexp
Identifier naming checks won't be enforced for static `constexpr`
variable names matching this regular expression.
```

(readability-identifier-naming-static-constexpr-variable-suffix)=

```{option} StaticConstexprVariableSuffix
When defined, the check will ensure static `constexpr` variable names
will add the suffix with the given value (regardless of casing).
```

(readability-identifier-naming-static-constexpr-variable-hungarian-prefix)=

```{option} StaticConstexprVariableHungarianPrefix
When enabled, the check ensures that the declared identifier will have a
Hungarian notation prefix based on the declared type.
```

For example using values of:

- [`StaticConstexprVariableCase`](#readability-identifier-naming-static-constexpr-variable-case) of `lower_case`
- [`StaticConstexprVariablePrefix`](#readability-identifier-naming-static-constexpr-variable-prefix) of `pre_`
- [`StaticConstexprVariableSuffix`](#readability-identifier-naming-static-constexpr-variable-suffix) of `_post`
- [`StaticConstexprVariableHungarianPrefix`](#readability-identifier-naming-static-constexpr-variable-hungarian-prefix) of `On`

Identifies and/or transforms static `constexpr` variable names as follows:

Before:

```c++
static unsigned constexpr MyConstexprStatic_array[] = {1, 2, 3};
```

After:

```c++
static unsigned constexpr pre_my_constexpr_static_array_post[] = {1, 2, 3};
```

(readability-identifier-naming-static-constant-case)=

```{option} StaticConstantCase
When defined, the check will ensure static constant names conform to the
selected casing.
```

(readability-identifier-naming-static-constant-prefix)=

```{option} StaticConstantPrefix
When defined, the check will ensure static constant names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-static-constant-ignored-regexp)=

```{option} StaticConstantIgnoredRegexp
Identifier naming checks won't be enforced for static constant names
matching this regular expression.
```

(readability-identifier-naming-static-constant-suffix)=

```{option} StaticConstantSuffix
When defined, the check will ensure static constant names will add the
suffix with the given value (regardless of casing).
```

(readability-identifier-naming-static-constant-hungarian-prefix)=

```{option} StaticConstantHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- [`StaticConstantCase`](#readability-identifier-naming-static-constant-case) of `lower_case`
- [`StaticConstantPrefix`](#readability-identifier-naming-static-constant-prefix) of `pre_`
- [`StaticConstantSuffix`](#readability-identifier-naming-static-constant-suffix) of `_post`
- [`StaticConstantHungarianPrefix`](#readability-identifier-naming-static-constant-hungarian-prefix) of `On`

Identifies and/or transforms static constant names as follows:

Before:

```c++
static unsigned const MyConstStatic_array[] = {1, 2, 3};
```

After:

```c++
static unsigned const pre_myconststatic_array_post[] = {1, 2, 3};
```

(readability-identifier-naming-static-variable-case)=

```{option} StaticVariableCase
When defined, the check will ensure static variable names conform to the
selected casing.
```

(readability-identifier-naming-static-variable-prefix)=

```{option} StaticVariablePrefix
When defined, the check will ensure static variable names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-static-variable-ignored-regexp)=

```{option} StaticVariableIgnoredRegexp
Identifier naming checks won't be enforced for static variable names
matching this regular expression.
```

(readability-identifier-naming-static-variable-suffix)=

```{option} StaticVariableSuffix
When defined, the check will ensure static variable names will add the
suffix with the given value (regardless of casing).
```

(readability-identifier-naming-static-variable-hungarian-prefix)=

```{option} StaticVariableHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- [`StaticVariableCase`](#readability-identifier-naming-static-variable-case) of `lower_case`
- [`StaticVariablePrefix`](#readability-identifier-naming-static-variable-prefix) of `pre_`
- [`StaticVariableSuffix`](#readability-identifier-naming-static-variable-suffix) of `_post`
- [`StaticVariableHungarianPrefix`](#readability-identifier-naming-static-variable-hungarian-prefix) of `On`

Identifies and/or transforms static variable names as follows:

Before:

```c++
static unsigned MyStatic_array[] = {1, 2, 3};
```

After:

```c++
static unsigned pre_mystatic_array_post[] = {1, 2, 3};
```

(readability-identifier-naming-struct-case)=

```{option} StructCase
When defined, the check will ensure struct names conform to the
selected casing.
```

(readability-identifier-naming-struct-prefix)=

```{option} StructPrefix
When defined, the check will ensure struct names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-struct-ignored-regexp)=

```{option} StructIgnoredRegexp
Identifier naming checks won't be enforced for struct names
matching this regular expression.
```

(readability-identifier-naming-struct-suffix)=

```{option} StructSuffix
When defined, the check will ensure struct names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- [`StructCase`](#readability-identifier-naming-struct-case) of `lower_case`
- [`StructPrefix`](#readability-identifier-naming-struct-prefix) of `pre_`
- [`StructSuffix`](#readability-identifier-naming-struct-suffix) of `_post`

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

(readability-identifier-naming-template-parameter-case)=

```{option} TemplateParameterCase
When defined, the check will ensure template parameter names conform to the
selected casing.
```

(readability-identifier-naming-template-parameter-prefix)=

```{option} TemplateParameterPrefix
When defined, the check will ensure template parameter names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-template-parameter-ignored-regexp)=

```{option} TemplateParameterIgnoredRegexp
Identifier naming checks won't be enforced for template parameter names
matching this regular expression.
```

(readability-identifier-naming-template-parameter-suffix)=

```{option} TemplateParameterSuffix
When defined, the check will ensure template parameter names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- [`TemplateParameterCase`](#readability-identifier-naming-template-parameter-case) of `lower_case`
- [`TemplateParameterPrefix`](#readability-identifier-naming-template-parameter-prefix) of `pre_`
- [`TemplateParameterSuffix`](#readability-identifier-naming-template-parameter-suffix) of `_post`

Identifies and/or transforms template parameter names as follows:

Before:

```c++
template <typename T> class Foo {};
```

After:

```c++
template <typename pre_t_post> class Foo {};
```

(readability-identifier-naming-template-template-parameter-case)=

```{option} TemplateTemplateParameterCase
When defined, the check will ensure template template parameter names conform to the
selected casing.
```

(readability-identifier-naming-template-template-parameter-prefix)=

```{option} TemplateTemplateParameterPrefix
When defined, the check will ensure template template parameter names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-template-template-parameter-ignored-regexp)=

```{option} TemplateTemplateParameterIgnoredRegexp
Identifier naming checks won't be enforced for template template parameter
names matching this regular expression.
```

(readability-identifier-naming-template-template-parameter-suffix)=

```{option} TemplateTemplateParameterSuffix
When defined, the check will ensure template template parameter names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- [`TemplateTemplateParameterCase`](#readability-identifier-naming-template-template-parameter-case) of `lower_case`
- [`TemplateTemplateParameterPrefix`](#readability-identifier-naming-template-template-parameter-prefix) of `pre_`
- [`TemplateTemplateParameterSuffix`](#readability-identifier-naming-template-template-parameter-suffix) of `_post`

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

(readability-identifier-naming-type-alias-case)=

```{option} TypeAliasCase
When defined, the check will ensure type alias names conform to the
selected casing.
```

(readability-identifier-naming-type-alias-prefix)=

```{option} TypeAliasPrefix
When defined, the check will ensure type alias names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-type-alias-ignored-regexp)=

```{option} TypeAliasIgnoredRegexp
Identifier naming checks won't be enforced for type alias names
matching this regular expression.
```

(readability-identifier-naming-type-alias-suffix)=

```{option} TypeAliasSuffix
When defined, the check will ensure type alias names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- [`TypeAliasCase`](#readability-identifier-naming-type-alias-case) of `lower_case`
- [`TypeAliasPrefix`](#readability-identifier-naming-type-alias-prefix) of `pre_`
- [`TypeAliasSuffix`](#readability-identifier-naming-type-alias-suffix) of `_post`

Identifies and/or transforms type alias names as follows:

Before:

```c++
using MY_STRUCT_TYPE = my_structure;
```

After:

```c++
using pre_my_struct_type_post = my_structure;
```

(readability-identifier-naming-typedef-case)=

```{option} TypedefCase
When defined, the check will ensure typedef names conform to the
selected casing.
```

(readability-identifier-naming-typedef-prefix)=

```{option} TypedefPrefix
When defined, the check will ensure typedef names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-typedef-ignored-regexp)=

```{option} TypedefIgnoredRegexp
Identifier naming checks won't be enforced for typedef names
matching this regular expression.
```

(readability-identifier-naming-typedef-suffix)=

```{option} TypedefSuffix
When defined, the check will ensure typedef names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- [`TypedefCase`](#readability-identifier-naming-typedef-case) of `lower_case`
- [`TypedefPrefix`](#readability-identifier-naming-typedef-prefix) of `pre_`
- [`TypedefSuffix`](#readability-identifier-naming-typedef-suffix) of `_post`

Identifies and/or transforms typedef names as follows:

Before:

```c++
typedef int MYINT;
```

After:

```c++
typedef int pre_myint_post;
```

(readability-identifier-naming-typedef-inherit-anon-tag-config)=

```{option} TypedefInheritAnonTagConfig
When `true`, a typedef or type alias that provides the only name of
an otherwise unnamed tag, as in `typedef enum {} MyEnum;`, is checked
against the naming style configured for the kind of that tag
(`AbstractClass`, `Class`, `Enum`, `Struct` or `Union`, i.e.
[`EnumCase`](#readability-identifier-naming-enum-case), [`EnumPrefix`](#readability-identifier-naming-enum-prefix), [`EnumSuffix`](#readability-identifier-naming-enum-suffix) and
[`EnumIgnoredRegexp`](#readability-identifier-naming-enum-ignored-regexp) for an enum) rather than against the typedef
or type alias style. If that kind configures no case, prefix or suffix,
the typedef or type alias style still applies. Typedefs of named tags, of
other typedefs and of non-tag types are not affected. Default is `false`.
```

For example using values of:

- [`TypedefInheritAnonTagConfig`](#readability-identifier-naming-typedef-inherit-anon-tag-config) of `true`
- [`EnumCase`](#readability-identifier-naming-enum-case) of `CamelCase`
- [`TypedefCase`](#readability-identifier-naming-typedef-case) of `lower_case`

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

(readability-identifier-naming-type-template-parameter-case)=

```{option} TypeTemplateParameterCase
When defined, the check will ensure type template parameter names conform to the
selected casing.
```

(readability-identifier-naming-type-template-parameter-prefix)=

```{option} TypeTemplateParameterPrefix
When defined, the check will ensure type template parameter names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-type-template-parameter-ignored-regexp)=

```{option} TypeTemplateParameterIgnoredRegexp
Identifier naming checks won't be enforced for type template names
matching this regular expression.
```

(readability-identifier-naming-type-template-parameter-suffix)=

```{option} TypeTemplateParameterSuffix
When defined, the check will ensure type template parameter names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- [`TypeTemplateParameterCase`](#readability-identifier-naming-type-template-parameter-case) of `lower_case`
- [`TypeTemplateParameterPrefix`](#readability-identifier-naming-type-template-parameter-prefix) of `pre_`
- [`TypeTemplateParameterSuffix`](#readability-identifier-naming-type-template-parameter-suffix) of `_post`

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

(readability-identifier-naming-union-case)=

```{option} UnionCase
When defined, the check will ensure union names conform to the
selected casing.
```

(readability-identifier-naming-union-prefix)=

```{option} UnionPrefix
When defined, the check will ensure union names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-union-ignored-regexp)=

```{option} UnionIgnoredRegexp
Identifier naming checks won't be enforced for union names
matching this regular expression.
```

(readability-identifier-naming-union-suffix)=

```{option} UnionSuffix
When defined, the check will ensure union names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- [`UnionCase`](#readability-identifier-naming-union-case) of `lower_case`
- [`UnionPrefix`](#readability-identifier-naming-union-prefix) of `pre_`
- [`UnionSuffix`](#readability-identifier-naming-union-suffix) of `_post`

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

(readability-identifier-naming-value-template-parameter-case)=

```{option} ValueTemplateParameterCase
When defined, the check will ensure value template parameter names conform to the
selected casing.
```

(readability-identifier-naming-value-template-parameter-prefix)=

```{option} ValueTemplateParameterPrefix
When defined, the check will ensure value template parameter names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-value-template-parameter-ignored-regexp)=

```{option} ValueTemplateParameterIgnoredRegexp
Identifier naming checks won't be enforced for value template parameter
names matching this regular expression.
```

(readability-identifier-naming-value-template-parameter-suffix)=

```{option} ValueTemplateParameterSuffix
When defined, the check will ensure value template parameter names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- [`ValueTemplateParameterCase`](#readability-identifier-naming-value-template-parameter-case) of `lower_case`
- [`ValueTemplateParameterPrefix`](#readability-identifier-naming-value-template-parameter-prefix) of `pre_`
- [`ValueTemplateParameterSuffix`](#readability-identifier-naming-value-template-parameter-suffix) of `_post`

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

(readability-identifier-naming-variable-case)=

```{option} VariableCase
When defined, the check will ensure variable names conform to the
selected casing.
```

(readability-identifier-naming-variable-prefix)=

```{option} VariablePrefix
When defined, the check will ensure variable names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-variable-ignored-regexp)=

```{option} VariableIgnoredRegexp
Identifier naming checks won't be enforced for variable names
matching this regular expression.
```

(readability-identifier-naming-variable-suffix)=

```{option} VariableSuffix
When defined, the check will ensure variable names will add the
suffix with the given value (regardless of casing).
```

(readability-identifier-naming-variable-hungarian-prefix)=

```{option} VariableHungarianPrefix
When enabled, the check ensures that the declared identifier will
have a Hungarian notation prefix based on the declared type.
```

For example using values of:

- [`VariableCase`](#readability-identifier-naming-variable-case) of `lower_case`
- [`VariablePrefix`](#readability-identifier-naming-variable-prefix) of `pre_`
- [`VariableSuffix`](#readability-identifier-naming-variable-suffix) of `_post`
- [`VariableHungarianPrefix`](#readability-identifier-naming-variable-hungarian-prefix) of `On`

Identifies and/or transforms variable names as follows:

Before:

```c++
unsigned MyVariable;
```

After:

```c++
unsigned pre_myvariable_post;
```

(readability-identifier-naming-virtual-method-case)=

```{option} VirtualMethodCase
When defined, the check will ensure virtual method names conform to the
selected casing.
```

(readability-identifier-naming-virtual-method-prefix)=

```{option} VirtualMethodPrefix
When defined, the check will ensure virtual method names will add the
prefix with the given value (regardless of casing).
```

(readability-identifier-naming-virtual-method-ignored-regexp)=

```{option} VirtualMethodIgnoredRegexp
Identifier naming checks won't be enforced for virtual method names
matching this regular expression.
```

(readability-identifier-naming-virtual-method-suffix)=

```{option} VirtualMethodSuffix
When defined, the check will ensure virtual method names will add the
suffix with the given value (regardless of casing).
```

For example using values of:

- [`VirtualMethodCase`](#readability-identifier-naming-virtual-method-case) of `lower_case`
- [`VirtualMethodPrefix`](#readability-identifier-naming-virtual-method-prefix) of `pre_`
- [`VirtualMethodSuffix`](#readability-identifier-naming-virtual-method-suffix) of `_post`

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

- [`HungarianNotation.General.TreatStructAsClass`](#readability-identifier-naming-hungarian-notation-general-treat-struct-as-class)
- [`HungarianNotation.DerivedType.Array`](#readability-identifier-naming-hungarian-notation-derived-type-array)
- [`HungarianNotation.DerivedType.Pointer`](#readability-identifier-naming-hungarian-notation-derived-type-pointer)
- [`HungarianNotation.DerivedType.FunctionPointer`](#readability-identifier-naming-hungarian-notation-derived-type-function-pointer)
- [`HungarianNotation.CString.CharPointer`](#readability-identifier-naming-hungarian-notation-c-string-char-pointer)
- [`HungarianNotation.CString.CharArray`](#readability-identifier-naming-hungarian-notation-c-string-char-array)
- [`HungarianNotation.CString.WideCharPointer`](#readability-identifier-naming-hungarian-notation-c-string-wide-char-pointer)
- [`HungarianNotation.CString.WideCharArray`](#readability-identifier-naming-hungarian-notation-c-string-wide-char-array)
- [`HungarianNotation.PrimitiveType.*`](#readability-identifier-naming-hungarian-notation-primitive-type)
- [`HungarianNotation.UserDefinedType.*`](#readability-identifier-naming-hungarian-notation-user-defined-type)

(readability-identifier-naming-hungarian-notation-general-treat-struct-as-class)=

```{option} HungarianNotation.General.TreatStructAsClass
When defined, the check will treat naming of struct as a class.
Default is `false`.
```

(readability-identifier-naming-hungarian-notation-derived-type-array)=

```{option} HungarianNotation.DerivedType.Array
When defined, the check will ensure variable name will add the prefix with
the given string. Default is `a`.
```

(readability-identifier-naming-hungarian-notation-derived-type-pointer)=

```{option} HungarianNotation.DerivedType.Pointer
When defined, the check will ensure variable name will add the prefix with
the given string. Default is `p`.
```

(readability-identifier-naming-hungarian-notation-derived-type-function-pointer)=

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

(readability-identifier-naming-hungarian-notation-c-string-char-pointer)=

```{option} HungarianNotation.CString.CharPointer
When defined, the check will ensure variable name will add the prefix with
the given string. Default is `sz`.
```

(readability-identifier-naming-hungarian-notation-c-string-char-array)=

```{option} HungarianNotation.CString.CharArray
When defined, the check will ensure variable name will add the prefix with
the given string. Default is `sz`.
```

(readability-identifier-naming-hungarian-notation-c-string-wide-char-pointer)=

```{option} HungarianNotation.CString.WideCharPointer
When defined, the check will ensure variable name will add the prefix with
the given string. Default is `wsz`.
```

(readability-identifier-naming-hungarian-notation-c-string-wide-char-array)=

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

(readability-identifier-naming-hungarian-notation-primitive-type)=

```{option} HungarianNotation.PrimitiveType.*
When defined, the check will ensure variable name of involved primitive
types will add the prefix with the given string. The default prefixes are
defined in the default mapping table.
```

(readability-identifier-naming-hungarian-notation-user-defined-type)=

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
