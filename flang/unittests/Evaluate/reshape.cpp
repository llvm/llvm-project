#include "testing.h"
#include "flang/Runtime/descriptor.h"
#include "flang/Runtime/transformational.h"
#include <cinttypes>

using namespace Fortran::common;
using namespace Fortran::runtime;

int main() {
  static const SubscriptValue sourceExtent[]{2, 3, 4};
  auto source{Descriptor::Create(TypeCategory::Integer, sizeof(std::int32_t),
      nullptr, 3, sourceExtent, CFI_attribute_allocatable)};
  source->Check();
  MATCH(3, source->rank());
  MATCH(sizeof(std::int32_t), source->ElementBytes());
  TEST(source->IsAllocatable());
  TEST(!source->IsPointer());
  for (int j{0}; j < 3; ++j) {
    source->GetDimension(j).SetBounds(1, sourceExtent[j]);
  }
  TEST(source->Allocate() == CFI_SUCCESS);
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
  StaticDescriptor<1> orderDescriptor;
  Descriptor &order{orderDescriptor.descriptor()};
  static const std::int32_t orderData[]{1, 2};
  static const SubscriptValue orderExtent[]{2};
  order.Establish(TypeCategory::Integer, static_cast<int>(sizeof orderData[0]),
      const_cast<void *>(reinterpret_cast<const void *>(orderData)), 1,
      orderExtent, CFI_attribute_pointer);
  orderDescriptor.Check();
  order.Check();
  MATCH(1, order.rank());
  MATCH(2, order.GetDimension(0).Extent());

  auto result{Descriptor::Create(TypeCategory::Integer, sizeof(std::int32_t),
      nullptr, 2, nullptr, CFI_attribute_allocatable)};
  TEST(result.get() != nullptr);
  RTNAME(Reshape)(*result, *source, *shape, &pad, &order, __FILE__, __LINE__);
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

  return testing::Complete();
}
