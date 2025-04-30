#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test $(TEST)  ########

# compile free-format .f* test 

build:
	@$(RM) $(TEST).$(EXE) core.* *.exe
	@echo ------------------------------------ building compile-only test $@
	$(FC) -Mfree -c $(FFLAGS) $(SRC)/sources/$(TEST).f* -o $(TEST).$(OBJX)
	@echo PASS

run: ;

verify: ;

