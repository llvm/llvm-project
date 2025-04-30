#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
par04: par04.$(OBJX)
	@echo ------------ executing test $@
	-$(RUN4) ./a.$(EXESUFFIX) $(LOG)
par04.$(OBJX): $(SRC)/par04.f check.$(OBJX)
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC)/par04.f
	@$(RM) ./a.$(EXESUFFIX)
	-$(FC) $(LDFLAGS) par04.$(OBJX) check.$(OBJX) $(LIBS) -o a.$(EXESUFFIX)
build: par04
run: ;
