#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
par03: par03.$(OBJX)
	@echo ------------ executing test $@
	-$(RUN4) ./a.$(EXESUFFIX) $(LOG)
par03.$(OBJX): $(SRC)/par03.f check.$(OBJX)
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC)/par03.f
	@$(RM) ./a.$(EXESUFFIX)
	-$(FC) $(LDFLAGS) par03.$(OBJX) check.$(OBJX) $(LIBS) -o a.$(EXESUFFIX)
build: par03
run: ;
