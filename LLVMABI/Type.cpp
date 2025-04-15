#include "Type.h"
#include <iostream>
#include <string>

void ABIBuiltinType::dump() const {
    std::cout << "BuiltinType: ";
    switch (kind) {
        case Void: std::cout << "Void\n"; break;
        case Bool: std::cout << "Bool\n"; break;
        case Integer: std::cout << "Integer\n"; break;
        case SignedInteger: std::cout << "SignedInteger\n"; break;
        case UnsignedInteger: std::cout << "UnsignedInteger\n"; break;
        case Int128: std::cout << "Int128\n"; break;
        case FloatingPoint: std::cout << "FloatingPoint\n"; break;
    }
}

void ABIRecordType::dump() const {
    std::cout << "RecordType: " << RecordName << "\n";
    std::cout << "Alignment: " << AlignmentInBits << " bits\n";
    for (const auto &F : Fields) {
        std::cout << "  Field: " << F.Name
                  << ", Offset: " << F.OffsetInBits << " bits"
                  << ", TypeClass: " << F.FieldType->getTypeClass()
                  << "\n";
    }
}
