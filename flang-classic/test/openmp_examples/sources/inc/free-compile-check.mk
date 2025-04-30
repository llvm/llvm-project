#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# Use this makefile when expecting compilation failures as identified
# in the free-format source (.f*). Do not exit with compilation failure;
# use the compilation check tool to validate the expected errors.

build:
	@$(RM) $(TEST).$(EXE) core.* *.exe
	@echo ------------------------------------ building compile-only test $@
	-$(FC) -Mfree -c $(FFLAGS) $(SRC)/sources/$(TEST).f* -o $(TEST).$(OBJX) > $(TEST).log 2>&1

run: ;

verify:
	$(COMP_CHECK) $(SRC)/sources/$(TEST).f* $(TEST).log $(FC)
