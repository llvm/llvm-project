#include <ranges>



void test(){
    struct R {
        int* begin() const{reurn nullptr;};
        int* end() const{return nullptr;};
    
        operator int() const { return 0; }
      };
      (void)std::ranges::to<int>(R{});
        //expected-error-re@*:* {{static assertion failed{{.*}}The target must be a class type}}
      (void)(R{} | std::ranges::to<int>());
        //expected-error-re@*:* {{static assertion failed{{.*}}The target must be a class type}}
    
}