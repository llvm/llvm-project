#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test array_constructor_nested_implied_do  ########

build: $(SRC)/$(TEST).f90
	@echo ------------------------------------ building test $(TEST)
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/$(TEST).f90 -o $(TEST).$(OBJX) > $(TEST).rslt 2>&1
	-$(FC) $(FFLAGS) $(LDFLAGS) $(TEST).$(OBJX) check.$(OBJX) $(LIBS) -o $(TEST).$(EXESUFFIX)

run:
	@echo ------------------------------------ executing test $(TEST)
	./$(TEST).$(EXESUFFIX)

verify: $(TEST).rslt
	@echo ------------------------------------ verifying test $(TEST)
	$(COMP_CHECK) $(SRC)/$(TEST).f90 $(TEST).rslt $(FC)
