#pragma once

#include "utils.h"

std::unique_ptr<CompilationDatabase> getCompilationDatabase(fs::path buildPath);
