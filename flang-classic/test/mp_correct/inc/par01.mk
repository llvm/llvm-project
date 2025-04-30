#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
par01: par01.$(OBJX)
	@echo ------------ executing test $@
	-$(RUN4) ./a.$(EXESUFFIX) $(LOG)
par01.$(OBJX): $(SRC)/par01.f check.$(OBJX)
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC)/par01.f
	@$(RM) ./a.$(EXESUFFIX)
	-$(FC) $(LDFLAGS) par01.$(OBJX) check.$(OBJX) $(LIBS) -o a.$(EXESUFFIX)
build: par01
run: ;
