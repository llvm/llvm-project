#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test $(TEST)  ########

# compile, link, run fixed-format .f test

build:
	@$(RM) $(TEST).$(EXE) core.* *.exe
	@echo ------------------------------------ building test $@
	$(FC) -Mfixed -c $(FFLAGS) $(SRC)/sources/$(TEST).f -o $(TEST).$(OBJX)
	$(FC) $(TEST).$(OBJX) $(LDFLAGS) -o $(TEST).$(EXE)

run:
	@echo ------------------------------------ executing test $(TEST)
	$(RUN4) $(TEST).$(EXE)
	@echo PASS

verify: ;

