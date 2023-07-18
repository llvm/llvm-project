// RUN: %clang_cc1 -fobjc-arc -fobjc-runtime-has-weak -Wnonnull %s -verify
//rdar://19211059

@interface NSObject @end

@interface Base : NSObject
- (nonnull id)bad:(nullable id)obj; // expected-note 2 {{previous declaration is here}}
- (nullable id)notAsBad:(nonnull id)obj;
@end

@interface Sub : Base
- (nullable id)bad:(nonnull id)obj; // expected-warning {{conflicting nullability specifier on return types, 'nullable' conflicts with existing specifier 'nonnull'}} \
                                    // expected-warning {{conflicting nullability specifier on parameter types, 'nonnull' conflicts with existing specifier 'nullable'}}
- (nonnull id)notAsBad:(nullable id)obj;
@end
