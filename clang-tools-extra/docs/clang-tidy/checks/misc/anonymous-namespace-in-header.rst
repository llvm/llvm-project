.. title:: clang-tidy - anonymous-namespace-in-header

anonymous-namespace-in-header
=============================

Finds anonymous namespaces in headers. Anonymous namespaces in headers can lead to
ODR (One Definition Rule) violations, as each translation unit including the header
will have its own unique version of the entities declared within the anonymous
namespace. This can cause issues when linking, as the linker may see multiple
definitions of the same entity, leading to unexpected behavior or linker errors.