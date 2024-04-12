#pragma once

#include "utils.h"

std::unique_ptr<CompilationDatabase> getCompilationDatabase(fs::path buildPath);

void dumpCompileCommand(const CompileCommand &cmd);
void dumpCompilationDatabase(const CompilationDatabase &cb);
