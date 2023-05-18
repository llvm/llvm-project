// Exercise some template issues.  Should not produce errors.

// Forward declaration.
template<class T> class TemplateClass;

// Full declaration.
template<class T>class TemplateClass {
public:
  TemplateClass() {}
private:
  T Member;
};

// Template alias.
template<class T> using TemplateClassAlias = TemplateClass<T>;
