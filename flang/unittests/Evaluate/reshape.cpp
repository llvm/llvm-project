#include "testing.h"
#include "../../runtime/descriptor.h"
#include "../../runtime/transformational.h"
#include <cinttypes>

using namespace Fortran::common;
using namespace Fortran::runtime;

int main() {
  static const SubscriptValue ones[]{1, 1, 1};
  static const SubscriptValue sourceExtent[]{2, 3, 4};
  auto source{Descriptor::Create(TypeCategory::Integer, sizeof(std::int32_t),
      nullptr, 3, sourceExtent, CFI_attribute_allocatable)};
  source->Check();
  MATCH(3, source->rank());
  MATCH(sizeof(std::int32_t), source->ElementBytes());
  TEST(source->IsAllocatable());
  TEST(!source->IsPointer());
  TEST(source->Allocate(ones, sourceExtent) == CFI_SUCCESS);
  TEST(source->IsAllocated());
  MATCH(2, source->GetDimension(0).Extent());
  MATCH(3, source->GetDimension(1).Extent());
  MATCH(4, source->GetDimension(2).Extent());
  MATCH(24, source->Elements());
  for (std::size_t j{0}; j < 24; ++j) {
    *source->OffsetElement<std::int32_t>(j * sizeof(std::int32_t)) = j;
  }

  static const std::int16_t shapeData[]{8, 4};
  static const SubscriptValue shapeExtent{2};
  auto shape{Descriptor::Create(TypeCategory::Integer,
      static_cast<int>(sizeof shapeData[0]),
      const_cast<void *>(reinterpret_cast<const void *>(shapeData)), 1,
      &shapeExtent, CFI_attribute_pointer)};
  shape->Check();
  MATCH(1, shape->rank());
  MATCH(2, shape->GetDimension(0).Extent());
  TEST(shape->IsPointer());
  TEST(!shape->IsAllocatable());

  StaticDescriptor<3> padDescriptor;
  Descriptor &pad{padDescriptor.descriptor()};
  static const std::int32_t padData[]{24, 25, 26, 27, 28, 29, 30, 31};
  static const SubscriptValue padExtent[]{2, 2, 3};
  pad.Establish(TypeCategory::Integer, static_cast<int>(sizeof padData[0]),
      const_cast<void *>(reinterpret_cast<const void *>(padData)), 3, padExtent,
      CFI_attribute_pointer);
  padDescriptor.Check();
  pad.Check();
  MATCH(3, pad.rank());
  MATCH(2, pad.GetDimension(0).Extent());
  MATCH(2, pad.GetDimension(1).Extent());
  MATCH(3, pad.GetDimension(2).Extent());

  auto result{
      RTNAME(Reshape)(*source, *shape, &pad, nullptr, __FILE__, __LINE__)};
  TEST(result.get() != nullptr);
  result->Check();
  MATCH(sizeof(std::int32_t), result->ElementBytes());
  MATCH(2, result->rank());
  TEST(result->type().IsInteger());
  for (std::int32_t j{0}; j < 32; ++j) {
    MATCH(j, *result->OffsetElement<std::int32_t>(j * sizeof(std::int32_t)));
  }
  for (std::int32_t j{0}; j < 32; ++j) {
    SubscriptValue ss[2]{1 + (j % 8), 1 + (j / 8)};
    MATCH(j, *result->Element<std::int32_t>(ss));
  }

  // TODO: test ORDER=

  return testing::Complete();
}
