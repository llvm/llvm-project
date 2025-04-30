#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

$(TEST): $(TEST).$(OBJX)
	@echo ------------ executing test $@
	-$(RUN1) ./$(TEST).$(EXESUFFIX) $(LOG)

$(TEST).$(OBJX): $(SRC)/src/$(TEST).f90 check.$(OBJX)
	@echo ------------ building test $@
	@echo $(FLAGS)
	-$(FC) -c $(MPFLAGS) $(OPT) -I$(SRC) $(SRC)/src/$(TEST).f90
	@$(RM) ./$(TEST).$(EXESUFFIX)
	-$(FC) $(MPFLAGS) $(OPT) $(TEST).$(OBJX) check.$(OBJX) $(LIBS) -o $(TEST).$(EXESUFFIX)

build: $(TEST)
run: ;
