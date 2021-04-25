#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import filecmp
import shutil
import argparse

class Generator(object):

    implementationContent = ''

    RefClades = {"DeclarationNameInfo",
        "NestedNameSpecifierLoc",
        "TemplateArgumentLoc",
        "TypeLoc"}

    def __init__(self, templateClasses):
        self.templateClasses = templateClasses

    def GeneratePrologue(self):

        self.implementationContent += \
            """
/*===- Generated file -------------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Introspection of available AST node SourceLocations                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace clang {
namespace tooling {

using LocationAndString = SourceLocationMap::value_type;
using RangeAndString = SourceRangeMap::value_type;

bool NodeIntrospection::hasIntrospectionSupport() { return true; }

struct RecursionPopper
{
    RecursionPopper(std::vector<clang::TypeLoc> &TypeLocRecursionGuard)
    :  TLRG(TypeLocRecursionGuard)
    {

    }

    ~RecursionPopper()
    {
    TLRG.pop_back();
    }

private:
std::vector<clang::TypeLoc> &TLRG;
};
"""

    def GenerateBaseGetLocationsDeclaration(self, CladeName):
        InstanceDecoration = "*"
        if CladeName in self.RefClades:
            InstanceDecoration = "&"

        self.implementationContent += \
            """
void GetLocationsImpl(SharedLocationCall const& Prefix,
    clang::{0} const {1}Object, SourceLocationMap &Locs,
    SourceRangeMap &Rngs,
    std::vector<clang::TypeLoc> &TypeLocRecursionGuard);
""".format(CladeName, InstanceDecoration)

    def GenerateSrcLocMethod(self,
            ClassName, ClassData, CreateLocalRecursionGuard):

        NormalClassName = ClassName
        RecursionGuardParam = ('' if CreateLocalRecursionGuard else \
            ', std::vector<clang::TypeLoc>& TypeLocRecursionGuard')

        if "templateParms" in ClassData:
            TemplatePreamble = "template <typename "
            ClassName += "<"
            First = True
            for TA in ClassData["templateParms"]:
                if not First:
                    ClassName += ", "
                    TemplatePreamble += ", typename "

                First = False
                ClassName += TA
                TemplatePreamble += TA

            ClassName += ">"
            TemplatePreamble += ">\n";
            self.implementationContent += TemplatePreamble

        self.implementationContent += \
            """
static void GetLocations{0}(SharedLocationCall const& Prefix,
    clang::{1} const &Object,
    SourceLocationMap &Locs, SourceRangeMap &Rngs {2})
{{
""".format(NormalClassName, ClassName, RecursionGuardParam)

        if 'sourceLocations' in ClassData:
            for locName in ClassData['sourceLocations']:
                self.implementationContent += \
                    """
  Locs.insert(LocationAndString(Object.{0}(),
    llvm::makeIntrusiveRefCnt<LocationCall>(Prefix, "{0}")));
""".format(locName)

            self.implementationContent += '\n'

        if 'sourceRanges' in ClassData:
            for rngName in ClassData['sourceRanges']:
                self.implementationContent += \
                    """
  Rngs.insert(RangeAndString(Object.{0}(),
    llvm::makeIntrusiveRefCnt<LocationCall>(Prefix, "{0}")));
""".format(rngName)

            self.implementationContent += '\n'

        if 'typeLocs' in ClassData or 'typeSourceInfos' in ClassData \
                or 'nestedNameLocs' in ClassData \
                or 'declNameInfos' in ClassData:
            if CreateLocalRecursionGuard:
                self.implementationContent += \
                    'std::vector<clang::TypeLoc> TypeLocRecursionGuard;\n'

            self.implementationContent += '\n'

            if 'typeLocs' in ClassData:
                for typeLoc in ClassData['typeLocs']:

                    self.implementationContent += \
                        """
              if (Object.{0}()) {{
                GetLocationsImpl(
                    llvm::makeIntrusiveRefCnt<LocationCall>(Prefix, "{0}"),
                    Object.{0}(), Locs, Rngs, TypeLocRecursionGuard);
                }}
              """.format(typeLoc)

            self.implementationContent += '\n'
            if 'typeSourceInfos' in ClassData:
                for tsi in ClassData['typeSourceInfos']:
                    self.implementationContent += \
                        """
              if (Object.{0}()) {{
                GetLocationsImpl(llvm::makeIntrusiveRefCnt<LocationCall>(
                    llvm::makeIntrusiveRefCnt<LocationCall>(Prefix, "{0}",
                        LocationCall::ReturnsPointer), "getTypeLoc"),
                    Object.{0}()->getTypeLoc(), Locs, Rngs, TypeLocRecursionGuard);
                    }}
              """.format(tsi)

                self.implementationContent += '\n'

            if 'nestedNameLocs' in ClassData:
                for NN in ClassData['nestedNameLocs']:
                    self.implementationContent += \
                        """
              if (Object.{0}())
                GetLocationsImpl(
                    llvm::makeIntrusiveRefCnt<LocationCall>(Prefix, "{0}"),
                    Object.{0}(), Locs, Rngs, TypeLocRecursionGuard);
              """.format(NN)

            if 'declNameInfos' in ClassData:
                for declName in ClassData['declNameInfos']:

                    self.implementationContent += \
                        """
                      GetLocationsImpl(
                          llvm::makeIntrusiveRefCnt<LocationCall>(Prefix, "{0}"),
                          Object.{0}(), Locs, Rngs, TypeLocRecursionGuard);
                      """.format(declName)

        self.implementationContent += '}\n'

    def GenerateFiles(self, OutputFile):
        with open(os.path.join(os.getcwd(),
                  OutputFile), 'w') as f:
            f.write(self.implementationContent)

    def GenerateBaseGetLocationsFunction(self, ASTClassNames,
            ClassEntries, CladeName, InheritanceMap,
            CreateLocalRecursionGuard):

        MethodReturnType = 'NodeLocationAccessors'
        InstanceDecoration = "*"
        if CladeName in self.RefClades:
            InstanceDecoration = "&"

        Signature = \
            'GetLocations(clang::{0} const {1}Object)'.format(
                CladeName, InstanceDecoration)
        ImplSignature = \
            """
    GetLocationsImpl(SharedLocationCall const& Prefix,
        clang::{0} const {1}Object, SourceLocationMap &Locs,
        SourceRangeMap &Rngs,
        std::vector<clang::TypeLoc> &TypeLocRecursionGuard)
    """.format(CladeName, InstanceDecoration)

        self.implementationContent += 'void {0} {{ '.format(ImplSignature)

        if CladeName == "TypeLoc":
            self.implementationContent += 'if (Object.isNull()) return;'

            self.implementationContent += \
                """
            if (llvm::find(TypeLocRecursionGuard, Object) != TypeLocRecursionGuard.end())
              return;
            TypeLocRecursionGuard.push_back(Object);
            RecursionPopper RAII(TypeLocRecursionGuard);
                """

        RecursionGuardParam = ''
        if not CreateLocalRecursionGuard:
            RecursionGuardParam = ', TypeLocRecursionGuard'

        ArgPrefix = '*'
        if CladeName in self.RefClades:
            ArgPrefix = ''
        self.implementationContent += \
            'GetLocations{0}(Prefix, {1}Object, Locs, Rngs {2});'.format(
                CladeName, ArgPrefix, RecursionGuardParam)

        if CladeName == "TypeLoc":
            self.implementationContent += \
                '''
        if (auto QTL = Object.getAs<clang::QualifiedTypeLoc>()) {
            auto Dequalified = QTL.getNextTypeLoc();
            return GetLocationsImpl(llvm::makeIntrusiveRefCnt<LocationCall>(Prefix, "getNextTypeLoc"),
                                Dequalified,
                                Locs,
                                Rngs,
                                TypeLocRecursionGuard);
        }'''

        for ASTClassName in ASTClassNames:
            if ASTClassName in self.templateClasses:
                continue
            if ASTClassName == CladeName:
                continue
            if CladeName != "TypeLoc":
                self.implementationContent += \
                """
if (auto Derived = llvm::dyn_cast<clang::{0}>(Object)) {{
  GetLocations{0}(Prefix, *Derived, Locs, Rngs {1});
}}
""".format(ASTClassName, RecursionGuardParam)
                continue

            self.GenerateBaseTypeLocVisit(ASTClassName, ClassEntries,
                RecursionGuardParam, InheritanceMap)

        self.implementationContent += '}'

        self.implementationContent += \
            """
{0} NodeIntrospection::{1} {{
  NodeLocationAccessors Result;
  SharedLocationCall Prefix;
  std::vector<clang::TypeLoc> TypeLocRecursionGuard;

  GetLocationsImpl(Prefix, Object, Result.LocationAccessors,
                   Result.RangeAccessors, TypeLocRecursionGuard);
""".format(MethodReturnType, Signature)

        self.implementationContent += 'return Result; }'

    def GenerateBaseTypeLocVisit(self, ASTClassName, ClassEntries,
            RecursionGuardParam, InheritanceMap):
        CallPrefix = 'Prefix'
        if ASTClassName != 'TypeLoc':
            CallPrefix = \
                '''llvm::makeIntrusiveRefCnt<LocationCall>(Prefix,
                    "getAs<clang::{0}>", LocationCall::IsCast)
                '''.format(ASTClassName)

        if ASTClassName in ClassEntries:

            self.implementationContent += \
            """
            if (auto ConcreteTL = Object.getAs<clang::{0}>())
              GetLocations{1}({2}, ConcreteTL, Locs, Rngs {3});
            """.format(ASTClassName, ASTClassName,
                       CallPrefix, RecursionGuardParam)

        if ASTClassName in InheritanceMap:
            for baseTemplate in self.templateClasses:
                if baseTemplate in InheritanceMap[ASTClassName]:
                    self.implementationContent += \
                    """
    if (auto ConcreteTL = Object.getAs<clang::{0}>())
      GetLocations{1}({2}, ConcreteTL, Locs, Rngs {3});
    """.format(InheritanceMap[ASTClassName], baseTemplate,
            CallPrefix, RecursionGuardParam)


    def GenerateDynNodeVisitor(self, CladeNames):
        MethodReturnType = 'NodeLocationAccessors'

        Signature = \
            'GetLocations(clang::DynTypedNode const &Node)'

        self.implementationContent += MethodReturnType \
            + ' NodeIntrospection::' + Signature + '{'

        for CladeName in CladeNames:
            if CladeName == "DeclarationNameInfo":
                continue
            self.implementationContent += \
                """
    if (const auto *N = Node.get<{0}>())
    """.format(CladeName)
            ArgPrefix = ""
            if CladeName in self.RefClades:
                ArgPrefix = "*"
            self.implementationContent += \
            """
      return GetLocations({0}const_cast<{1} *>(N));""".format(ArgPrefix, CladeName)

        self.implementationContent += '\nreturn {}; }'

    def GenerateEpilogue(self):

        self.implementationContent += '''
  }
}
'''

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--json-input-path',
                      help='Read API description from FILE', metavar='FILE')
    parser.add_argument('--output-file', help='Generate output in FILEPATH',
                      metavar='FILEPATH')
    parser.add_argument('--use-empty-implementation',
                      help='Generate empty implementation',
                      action="store", type=int)
    parser.add_argument('--empty-implementation',
                      help='Copy empty implementation from FILEPATH',
                      action="store", metavar='FILEPATH')

    options = parser.parse_args()

    use_empty_implementation = options.use_empty_implementation

    if (not use_empty_implementation
            and not os.path.exists(options.json_input_path)):
        use_empty_implementation = True

    if not use_empty_implementation:
        with open(options.json_input_path) as f:
            jsonData = json.load(f)

        if not 'classesInClade' in jsonData or not jsonData["classesInClade"]:
            use_empty_implementation = True

    if use_empty_implementation:
        if not os.path.exists(options.output_file) or \
                not filecmp.cmp(options.empty_implementation, options.output_file):
            shutil.copyfile(options.empty_implementation, options.output_file)
        sys.exit(0)

    templateClasses = []
    for (ClassName, ClassAccessors) in jsonData['classEntries'].items():
        if "templateParms" in ClassAccessors:
            templateClasses.append(ClassName)

    g = Generator(templateClasses)

    g.GeneratePrologue()

    for (CladeName, ClassNameData) in jsonData['classesInClade'].items():
        g.GenerateBaseGetLocationsDeclaration(CladeName)

    def getCladeName(ClassName):
      for (CladeName, ClassNameData) in jsonData['classesInClade'].items():
        if ClassName in ClassNameData:
          return CladeName

    for (ClassName, ClassAccessors) in jsonData['classEntries'].items():
        cladeName = getCladeName(ClassName)
        g.GenerateSrcLocMethod(
            ClassName, ClassAccessors,
            cladeName not in Generator.RefClades)

    for (CladeName, ClassNameData) in jsonData['classesInClade'].items():
        g.GenerateBaseGetLocationsFunction(
            ClassNameData,
            jsonData['classEntries'],
            CladeName,
            jsonData["classInheritance"],
            CladeName not in Generator.RefClades)

    g.GenerateDynNodeVisitor(jsonData['classesInClade'].keys())

    g.GenerateEpilogue()

    g.GenerateFiles(options.output_file)

if __name__ == '__main__':
    main()
