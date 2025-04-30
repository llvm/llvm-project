#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
par09: par09.$(OBJX)
	@echo ------------ executing test $@
	-$(RUN4) ./a.$(EXESUFFIX) $(LOG)
par09.$(OBJX): $(SRC)/par09.f check.$(OBJX)
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC)/par09.f
	@$(RM) ./a.$(EXESUFFIX)
	-$(FC) $(LDFLAGS) par09.$(OBJX) check.$(OBJX) $(LIBS) -o a.$(EXESUFFIX)
build: par09
run: ;
