#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
si01: si01.$(OBJX)
	@echo ------------ executing test $@
	-$(RUN2) ./a.$(EXESUFFIX) $(LOG)
si01.$(OBJX): $(SRC)/si01.f check.$(OBJX)
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC)/si01.f
	@$(RM) ./a.$(EXESUFFIX)
	-$(FC) $(LDFLAGS) si01.$(OBJX) check.$(OBJX) $(LIBS) -o a.$(EXESUFFIX)
build: si01
run: ;
