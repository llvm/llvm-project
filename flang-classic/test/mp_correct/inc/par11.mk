#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
par11: par11.$(OBJX)
	@echo ------------ executing test $@
	-$(RUN4) ./a.$(EXESUFFIX) $(LOG)
par11.$(OBJX): $(SRC)/par11.f check.$(OBJX)
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC)/par11.f
	@$(RM) ./a.$(EXESUFFIX)
	-$(FC) $(LDFLAGS) par11.$(OBJX) check.$(OBJX) $(LIBS) -o a.$(EXESUFFIX)
build: par11
run: ;
