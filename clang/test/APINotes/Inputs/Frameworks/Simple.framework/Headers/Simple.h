#ifndef SIMPLE_H
#define SIMPLE_H

__attribute__((__objc_root__))
@interface I
@property(class, nonatomic, readonly) id nonnullProperty;
@property(class, nonatomic, readonly) id nonnullNewProperty;

@property(class, nonatomic, readonly) id optionalProperty;
@property(class, nonatomic, readonly) id optionalNewProperty;

@property(nonatomic, readonly) id unspecifiedProperty;
@property(nonatomic, readonly) id unspecifiedNewProperty;

@property(nonatomic, readonly) id scalarProperty;
@property(nonatomic, readonly) id scalarNewProperty;
@end

#endif
