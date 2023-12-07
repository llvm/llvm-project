// RUN: %clang_cc1 -emit-llvm-only -std=c++11 -fblocks -o - -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 %s

@class DYFuture;
@interface NSCache
- (void)setObject:(id)obj forKey:(id)key;
@end

template <typename T>
class ResourceManager
{
public:
 ~ResourceManager();
 DYFuture* XXX();
 NSCache* _spDeviceCache;
};

template <typename T>
DYFuture* ResourceManager<T>::XXX()
{
 ^ {
   [_spDeviceCache setObject:0 forKey:0];
  }();

 return 0;
}

struct AnalyzerBaseObjectTypes { };

void FUNC()
{
    ResourceManager<AnalyzerBaseObjectTypes> *rm;
    ^(void) { rm->XXX(); }();
}

namespace PR9982 {
  template<typename T> struct Curry;

  template<typename R, typename Arg0, typename Arg1, typename Arg2>
    struct Curry<R (^)(Arg0, Arg1, Arg2)>
    {
      typedef R (^FType)(Arg0, Arg1, Arg2);
    
    Curry(FType _f) : f(_f) {}
      ~Curry() {;}
    
      R (^(^operator()(Arg0 a))(Arg1))(Arg2) 
      { 
        auto block = ^(Arg1 b) {
          auto inner_block = ^(Arg2 c) {
            return f(a, b, c);
          };
          return inner_block; 
        };
        return block;
      }
    
    private:
      FType f;
    };

  auto add = ^(int a, int b, int c)
    {
      return a + b + c;
    };

  void curry() {
    Curry<__decltype(add)> c = Curry<__decltype(add)>(add);
    auto t = c(1)(10)(100);
  }
}
