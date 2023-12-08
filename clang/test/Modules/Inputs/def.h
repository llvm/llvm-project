#include "def-include.h"



@interface A {
@public
  int ivar;
}
@end

@interface Def
- defMethod;
@end

#ifdef __cplusplus
class Def2 {
public:
  void func();
};

namespace Def3NS {
  class Def3 {
  public:
   void func();
  };
}
#endif
