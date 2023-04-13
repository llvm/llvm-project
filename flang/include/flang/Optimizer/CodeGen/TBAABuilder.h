//===-- TBAABuilder.h -- TBAA builder declarations --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_CODEGEN_TBAABUILDER_H
#define FORTRAN_OPTIMIZER_CODEGEN_TBAABUILDER_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace fir {

// TBAA builder provides mapping between FIR types and their TBAA type
// descriptors, and methods to populate that mapping during FIR to LLVM
// type conversion and to attach llvm.tbaa attributes to memory access
// instructions.
//
// TBAA type information is represented with LLVM::MetadataOp operation
// with specific symbol name `TBAABuilder::tbaaMetaOpName`. The basic
// TBAA tree used for Flang consists of the following nodes:
//   llvm.metadata @__flang_tbaa {
//     llvm.tbaa_root @root_0 {id = "Flang Type TBAA Root"}
//     llvm.tbaa_type_desc @type_desc_1 {id = "any access",
//                                       members = {<@root_0, 0>}}
//     llvm.tbaa_type_desc @type_desc_2 {id = "any data access",
//                                       members = {<@type_desc_1, 0>}}
//     llvm.tbaa_type_desc @type_desc_3 {id = "descriptor member",
//                                       members = {<@type_desc_1, 0>}}
//   }
//
// The `<any data access>` and `<descriptor member>` type descriptors
// are two sub-roots of the basic TBAA tree, and they allow representing
// box and non-box accesses, which can never alias in the current Flang
// implementation. The `<any access>` type descriptor is their common parent
// that can be used for representing accesses that may alias box and non-box
// accesses if an access cannot be classified strictly as box or non-box.
// In the current implementation `<any access>` is not used by TBAA access tags,
// because it is always known whether an operation accesses box or non-box.
//
// Given this basic TBAA tree structure, the box/descriptor types may
// be represented like this:
//   llvm.tbaa_type_desc @type_desc_4 {
//       id = "CFI_cdesc_t_dim0",
//       members = {<@type_desc_3, 0>, // base_addr
//                  <@type_desc_3, 8>, // elem_len
//                  <@type_desc_3, 16>, // version
//                  <@type_desc_3, 20>, // rank
//                  <@type_desc_3, 21>, // type
//                  <@type_desc_3, 22>, // attribute
//                  <@type_desc_3, 23>} // f18Addendum
//   }
//   llvm.tbaa_type_desc @type_desc_5 {
//       id = "CFI_cdesc_t_dim1",
//       members = {<@type_desc_3, 0>, // base_addr
//                  <@type_desc_3, 8>, // elem_len
//                  <@type_desc_3, 16>, // version
//                  <@type_desc_3, 20>, // rank
//                  <@type_desc_3, 21>, // type
//                  <@type_desc_3, 22>, // attribute
//                  <@type_desc_3, 23>, // f18Addendum
//                  <@type_desc_3, 24>, // dim[0].lower_bound
//                  <@type_desc_3, 32>, // dim[0].extent
//                  <@type_desc_3, 40>} // dim[0].sm
//   }
//   llvm.tbaa_type_desc @type_desc_6 {
//       id = "CFI_cdesc_t_dim2",
//       members = {<@type_desc_3, 0>, // base_addr
//                  <@type_desc_3, 8>, // elem_len
//                  <@type_desc_3, 16>, // version
//                  <@type_desc_3, 20>, // rank
//                  <@type_desc_3, 21>, // type
//                  <@type_desc_3, 22>, // attribute
//                  <@type_desc_3, 23>, // f18Addendum
//                  <@type_desc_3, 24>, // dim[0].lower_bound
//                  <@type_desc_3, 32>, // dim[0].extent
//                  <@type_desc_3, 40>, // dim[0].sm
//                  <@type_desc_3, 48>, // dim[1].lower_bound
//                  <@type_desc_3, 56>, // dim[1].extent
//                  <@type_desc_3, 64>} // dim[1].sm
//   }
// etc.
//
// Note that the TBAA type descriptors cannot represent array members
// of structures, so the `dim` array in the descriptor structure
// has to be represented as linear set of members.
//
// We can use the same TBAA type descriptor for all members of the F18
// descriptor structure, because the actual accesses of the F18 descriptor
// members will be disambiguated based on their offset off the beginning
// of the descriptor. Thus, all members have the same `<descriptor member>`
// type in the TBAA graph.
//
// The TBAA type descriptors have to be created during FIR to LLVM type
// conversion, so fir::LLVMTypeConverter has to provide the member offsets
// to TBAABuilder - the offsets must be computed based on the LLVM type
// to which the FIR type is being converted.
//
// TBAABuilder keeps a map between the FIR type and its TBAA type descriptor.
// The map is used when a TBAA tag needs to be attached to a memory accessing
// operation given the FIR types identifying the access's base and access type
// and the offset within the base type, e.g. an access of one dimensional
// descriptor's `base_addr` member may be defined by:
//   * base FIR type: !fir.box<!fir.array<?xf32>> - the resulting
//     access tag will use `<CFI_cdesc_t_dim1>` type descriptor for the base
//     type.
//   * access FIR type: <undefined> - all accesses within descriptors
//     are always represented with `<descriptor member>` type descriptor.
//   * offset:
//       llvm.getelementptr %arg0[0, 0] :
//           (!llvm.ptr<struct<(ptr<f32>, i64, i32, i8, i8, i8, i8,
//                              array<1 x array<3 x i64>>)>>) ->
//           !llvm.ptr<ptr<f32>>
//     The offset is computed based on the LLVM::GEPOp's indices and the LLVM
//     type layout.
//
// Detailed representation of the layout of the F18 descriptors is required
// to disambiguate accesses of the different members of the descriptors,
// e.g. a read of `base_addr` member (of one box) can never alias with
// a write of `rank` member (of another box).
//
// TODO: define handling of assumed-rank arrays' boxes (they can always
// be represented with a conservative tag:
//   < `<descriptor member>`, `<descriptor member>`, 0 >
// so that they alias with any other box accesses.
//
// The same representation can be used for user-defined types, though,
// strict type aliasing cannot be applied for Fortran programs without
// additional guarantees from the user. Fortran's storage association
// constructs provide a way to alias data of different types, so using
// TBAA would be incorrect, e.g.:
//   subroutine test()
//     real :: a
//     integer :: b
//     equivalence (a, b)
//     a = 1.0
//     call test2(b)
//   end subroutine test
//
// The store operation for `a = 1.0` has the base/access type `f32`,
// while a load from `b` inside `test2` will have base/access type
// `i32`. Due to the storage association the store and the load alias,
// so using the access types to create TBAA access tags may result
// in an incorrect result if `test2` was inlined. Moreover, in the scope
// of `test2` Flang is not able to indentify whether `b` is part
// of an equivalence.
//
// TBAA may still be applied for programs not using storage association
// for objects of different data types (e.g. under an opt-in compiler option).
//
// The initial implementation does not create detailed type descriptors
// for box types and always uses the conservative box access tag:
//   < `<descriptor member>`, `<descriptor member>`, 0 >
//
// Given the storage association, all non-box accesses are represented
// with the conservative data access tag:
//   < `<any data access>`, `<any data access>`, 0 >
class TBAABuilder {
public:
  TBAABuilder(mlir::ModuleOp module, bool applyTBAA);
  TBAABuilder(TBAABuilder const &) = delete;
  TBAABuilder &operator=(TBAABuilder const &) = delete;

  // Attach the llvm.tbaa attribute to the given memory accessing operation
  // based on the provided base/access FIR types and the GEPOp.
  void attachTBAATag(mlir::LLVM::AliasAnalysisOpInterface op,
                     mlir::Type baseFIRType, mlir::Type accessFIRType,
                     mlir::LLVM::GEPOp gep);

private:
  // Return unique string name based on `basename`.
  std::string getNewTBAANodeName(llvm::StringRef basename);

  // Find or create TBAATagOp operation (TBAA access tag) with the specified
  // components and return the symbol it defines.
  mlir::SymbolRefAttr getAccessTag(mlir::SymbolRefAttr baseTypeDesc,
                                   mlir::SymbolRefAttr accessTypeDesc,
                                   int64_t offset);
  // Returns symbol of TBAATagOp representing access tag:
  //   < <descriptor member>, <descriptor member>, 0 >
  mlir::SymbolRefAttr getAnyBoxAccessTag();
  // Returns symbol of TBAATagOp representing access tag:
  //   < <any data access>, <any data access>, 0 >
  mlir::SymbolRefAttr getAnyDataAccessTag();

  // Returns symbol of TBAATagOp representing access tag
  // described by the base and access FIR types and the LLVM::GepOp
  // representing the access in terms of the FIR types converted
  // to LLVM types. The base type must be derivative of fir::BaseBoxType.
  mlir::SymbolRefAttr getBoxAccessTag(mlir::Type baseFIRType,
                                      mlir::Type accessFIRType,
                                      mlir::LLVM::GEPOp gep);

  // Returns symbol of TBAATagOp representing access tag
  // described by the base and access FIR types and the LLVM::GepOp
  // representing the access in terms of the FIR types converted
  // to LLVM types. The FIR types must describe the "data" access,
  // i.e. not an access of any box/descriptor member.
  mlir::SymbolRefAttr getDataAccessTag(mlir::Type baseFIRType,
                                       mlir::Type accessFIRType,
                                       mlir::LLVM::GEPOp gep);

  // Set to true, if TBAA builder is active, otherwise, all public
  // methods are no-ops.
  bool enableTBAA;

  // LLVM::MetadataOp holding the TBAA operations.
  mlir::LLVM::MetadataOp tbaaMetaOp;
  // Symbol name of tbaaMetaOp.
  static constexpr llvm::StringRef tbaaMetaOpName = "__flang_tbaa";

  // Base names for TBAA operations:
  //   TBAARootMetadataOp:
  static constexpr llvm::StringRef kRootSymBasename = "root";
  //   TBAATypeDescriptorOp:
  static constexpr llvm::StringRef kTypeDescSymBasename = "type_desc";
  //   TBAATagOp:
  static constexpr llvm::StringRef kTagSymBasename = "tag";

  // Symbol defined by the LLVM::TBAARootMetadataOp identifying
  // Flang's TBAA root.
  mlir::SymbolRefAttr flangTBAARoot;
  // Identity string for Flang's TBAA root.
  static constexpr llvm::StringRef flangTBAARootId = "Flang Type TBAA Root";

  // Symbol defined by LLVM::TBAATypeDescriptorOp identifying
  // "any access".
  mlir::SymbolRefAttr anyAccessTypeDesc;
  // Identity string for "any access" type descriptor.
  static constexpr llvm::StringRef anyAccessTypeDescId = "any access";

  // Symbol defined by LLVM::TBAATypeDescriptorOp identifying
  // "any data access" (i.e. non-box memory access).
  mlir::SymbolRefAttr anyDataAccessTypeDesc;
  // Identity string for "any data access" type descriptor.
  static constexpr llvm::StringRef anyDataAccessTypeDescId = "any data access";

  // Symbol defined by LLVM::TBAATypeDescriptorOp identifying
  // "descriptor member" access, i.e. any access within the bounds
  // of a box/descriptor.
  mlir::SymbolRefAttr boxMemberTypeDesc;
  // Identity string for "descriptor member" type descriptor.
  static constexpr llvm::StringRef boxMemberTypeDescId = "descriptor member";

  // Counter for unique naming of TBAA operations' symbols.
  unsigned tbaaNodeCounter = 0;

  // Number of attached TBAA tags (used for debugging).
  unsigned tagAttachmentCounter = 0;

  // Mapping from a FIR type to the symbol defined by the corresponding
  // TBAATypeDescriptorOp. It must be populated during the type conversion.
  // Currently unused.
  llvm::DenseMap<mlir::Type, mlir::SymbolRefAttr> typeDescMap;

  // Each TBAA tag is a tuple of <baseTypeSym, accessTypeSym, offset>.
  // This map holds TBAATagOp symbol for each unique tuple.
  llvm::DenseMap<std::tuple<mlir::SymbolRefAttr, mlir::SymbolRefAttr, int64_t>,
                 mlir::SymbolRefAttr>
      tagsMap;
};

} // namespace fir

#endif // FORTRAN_OPTIMIZER_CODEGEN_TBAABUILDER_H
