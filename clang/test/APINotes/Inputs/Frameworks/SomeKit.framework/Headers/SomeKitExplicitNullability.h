@interface A(ExplicitNullabilityProperties)
@property (nonatomic, readwrite, retain, nonnull) A *explicitNonnullInstance;
@property (nonatomic, readwrite, retain, nullable) A *explicitNullableInstance;
@end
