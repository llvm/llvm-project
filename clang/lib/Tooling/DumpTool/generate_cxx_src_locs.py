#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import json

from optparse import OptionParser

parser = OptionParser()
parser.add_option('--json-input-path',
                  help='Read API description from FILE', metavar='FILE')
parser.add_option('--output-file', help='Generate output in FILEPATH',
                  metavar='FILEPATH')
parser.add_option('--empty-implementation', help='Generate empty implementation',
                  action="store", type="int", metavar='FILEPATH')

(options, args) = parser.parse_args()

if options.empty_implementation:
    with open(os.path.join(os.getcwd(),
              options.output_file), 'w') as f:
        f.write("""
namespace clang {
namespace tooling {

NodeLocationAccessors NodeIntrospection::GetLocations(clang::Stmt const *) {
  return {};
}
NodeLocationAccessors
NodeIntrospection::GetLocations(clang::DynTypedNode const &) {
  return {};
}
} // namespace tooling
} // namespace clang
""")
    sys.exit(0)

with open(options.json_input_path) as f:
    jsonData = json.load(f)


class Generator(object):

    implementationContent = ''

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
"""

    def GenerateBaseGetLocationsDeclaration(self, CladeName):
        self.implementationContent += \
            """
void GetLocationsImpl(std::shared_ptr<LocationCall> const& Prefix, clang::{0} const *Object, SourceLocationMap &Locs,
    SourceRangeMap &Rngs);
""".format(CladeName)

    def GenerateSrcLocMethod(self, ClassName, ClassData):

        self.implementationContent += \
            """
static void GetLocations{0}(std::shared_ptr<LocationCall> const& Prefix,
    clang::{0} const &Object,
    SourceLocationMap &Locs, SourceRangeMap &Rngs)
{{
""".format(ClassName)

        if 'sourceLocations' in ClassData:
            for locName in ClassData['sourceLocations']:
                self.implementationContent += \
                    """
  Locs.insert(LocationAndString(Object.{0}(), std::make_shared<LocationCall>(Prefix, "{0}")));
""".format(locName)

            self.implementationContent += '\n'

        if 'sourceRanges' in ClassData:
            for rngName in ClassData['sourceRanges']:
                self.implementationContent += \
                    """
  Rngs.insert(RangeAndString(Object.{0}(), std::make_shared<LocationCall>(Prefix, "{0}")));
""".format(rngName)

            self.implementationContent += '\n'

        self.implementationContent += '}\n'

    def GenerateFiles(self, OutputFile):
        with open(os.path.join(os.getcwd(),
                  OutputFile), 'w') as f:
            f.write(self.implementationContent)

    def GenerateTrivialBaseGetLocationsFunction(self, CladeName):
        MethodReturnType = 'NodeLocationAccessors'

        Signature = \
            'GetLocations(clang::{0} const *Object)'.format(CladeName)

        self.implementationContent += \
            '{0} NodeIntrospection::{1} {{ return {{}}; }}'.format(MethodReturnType,
                Signature)

    def GenerateBaseGetLocationsFunction(self, ASTClassNames, CladeName):

        MethodReturnType = 'NodeLocationAccessors'

        Signature = \
            'GetLocations(clang::{0} const *Object)'.format(CladeName)
        ImplSignature = \
            """
GetLocationsImpl(std::shared_ptr<LocationCall> const& Prefix,
    clang::{0} const *Object, SourceLocationMap &Locs,
    SourceRangeMap &Rngs)
""".format(CladeName)

        self.implementationContent += \
            'void {0} {{ GetLocations{1}(Prefix, *Object, Locs, Rngs);'.format(ImplSignature,
                CladeName)

        for ASTClassName in ASTClassNames:
            if ASTClassName != CladeName:
                self.implementationContent += \
                    """
if (auto Derived = llvm::dyn_cast<clang::{0}>(Object)) {{
  GetLocations{0}(Prefix, *Derived, Locs, Rngs);
}}
""".format(ASTClassName)

        self.implementationContent += '}'

        self.implementationContent += \
            """
{0} NodeIntrospection::{1} {{
  NodeLocationAccessors Result;
  std::shared_ptr<LocationCall> Prefix;

  GetLocationsImpl(Prefix, Object, Result.LocationAccessors,
                   Result.RangeAccessors);
""".format(MethodReturnType,
                Signature)

        self.implementationContent += 'return Result; }'

    def GenerateDynNodeVisitor(self, CladeNames):
        MethodReturnType = 'NodeLocationAccessors'

        Signature = \
            'GetLocations(clang::DynTypedNode const &Node)'

        self.implementationContent += MethodReturnType \
            + ' NodeIntrospection::' + Signature + '{'

        for CladeName in CladeNames:
            self.implementationContent += \
                """
    if (const auto *N = Node.get<{0}>())
      return GetLocations(const_cast<{0} *>(N));""".format(CladeName)

        self.implementationContent += 'return {}; }'

    def GenerateEpilogue(self):

        self.implementationContent += '''
  }
}
'''


g = Generator()

g.GeneratePrologue()

if 'classesInClade' in jsonData:
    for (CladeName, ClassNameData) in jsonData['classesInClade'].items():
        g.GenerateBaseGetLocationsDeclaration(CladeName)

    for (ClassName, ClassAccessors) in jsonData['classEntries'].items():
        if ClassAccessors:
            g.GenerateSrcLocMethod(ClassName, ClassAccessors)

    for (CladeName, ClassNameData) in jsonData['classesInClade'].items():
        g.GenerateBaseGetLocationsFunction(ClassNameData, CladeName)

    g.GenerateDynNodeVisitor(jsonData['classesInClade'].keys())
else:
    for CladeName in ['Stmt']:
        g.GenerateTrivialBaseGetLocationsFunction(CladeName)

    g.GenerateDynNodeVisitor([])

g.GenerateEpilogue()

g.GenerateFiles(options.output_file)
