struct DefinedInBImpl {
  void f();
  struct Inner {};
  friend void FoundByADL(DefinedInBImpl);
};

@import cxx_templates_common;
template struct TemplateInstantiationVisibility<char[1]>;
extern template struct TemplateInstantiationVisibility<char[2]>;
template<> struct TemplateInstantiationVisibility<char[3]> {};
extern TemplateInstantiationVisibility<char[4]>::type
    TemplateInstantiationVisibility_ImplicitInstantiation;
