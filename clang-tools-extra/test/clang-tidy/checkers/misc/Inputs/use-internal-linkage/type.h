#pragma once

enum EnumDeclaredInHeader : int;
struct StructDeclaredInHeader;
union UnionDeclaredInHeader;
class ClassDeclaredInHeader;

template <typename>
class TemplateDeclaredInHeader {};

extern template class TemplateDeclaredInHeader<char>;
