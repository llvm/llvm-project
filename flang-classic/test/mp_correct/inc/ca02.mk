#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
ca02: ca02.$(OBJX)
	@echo ------------ executing test $@
	-$(RUN1) ./a.$(EXESUFFIX) $(LOG)
	-$(RUN2) ./a.$(EXESUFFIX) $(LOG)
	-$(RUN4) ./a.$(EXESUFFIX) $(LOG)
ca02.$(OBJX): $(SRC)/ca02.f check.$(OBJX)
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC)/ca02.f
	@$(RM) ./a.$(EXESUFFIX)
	-$(FC) $(LDFLAGS) ca02.$(OBJX) check.$(OBJX) $(LIBS) -o a.$(EXESUFFIX)
build: ca02
run: ;
