// Include a file to ensure we have multiple sources.
#include "sample.h"

// This introduces a symbol, a reference and a relation.
struct Bar : public Foo {
  /// \brief This introduces an OverriddenBy relation by implementing Foo::Func.
  /// \details And it also introduces some doxygen!
  /// \param foo bar
  /// \warning !!!
  /// \note a note
  /// \return nothing
  void Func() override {}
};
