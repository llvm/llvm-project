#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

build: clean
	@echo ------------------------------------ building test $(TEST)
#	-@$(CP) $(SRC)/$(TEST).f90  .
	-$(FC) -c $(FCFLAGS) $(SRC)/$(TEST).f90 > $(TEST).rslt 2>&1

run:
	@echo ------------------------------------ test $(TEST) not expected to execute

verify:
	@echo ------------------------------------ verifying test $(TEST)
	$(COMP_CHECK) $(SRC)/$(TEST).f90 $(TEST).rslt $(FC)

clean:
	-@$(RM) $(TEST).rslt $(TEST).$(OBJX) *.mod $(TEST).$(EXE)

