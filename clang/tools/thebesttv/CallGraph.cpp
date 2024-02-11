#include "CallGraph.h"

std::map<std::string, std::set<std::string>>
    GenWholeProgramCallGraphVisitor::callGraph;

std::map<std::string, NamedLocation *>
    GenWholeProgramCallGraphVisitor::infoOfFunction;
