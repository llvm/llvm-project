#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
si00: si00.$(OBJX)
	@echo ------------ executing test $@
	-$(RUN2) ./a.$(EXESUFFIX) $(LOG)
si00.$(OBJX): $(SRC)/si00.f check.$(OBJX)
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC)/si00.f
	@$(RM) ./a.$(EXESUFFIX)
	-$(FC) $(LDFLAGS) si00.$(OBJX) check.$(OBJX) $(LIBS) -o a.$(EXESUFFIX)
build: si00
run: ;
