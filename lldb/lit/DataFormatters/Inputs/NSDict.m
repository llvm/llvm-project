#include <Foundation/Foundation.h>

int main(void)
{
  NSDictionary *emptyDictionary = [[NSDictionary alloc] init];
  NSMutableDictionary *mutableDictionary = [NSMutableDictionary dictionary];
  NSDictionary *dictionary = @{ @"key": @{} };
  return 0;
}
