#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
pa00: pa00.$(OBJX)
	@echo ------------ executing test $@
	-$(RUN2) ./a.$(EXESUFFIX) $(LOG)
pa00.$(OBJX): $(SRC)/pa00.f check.$(OBJX)
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC)/pa00.f
	@$(RM) ./a.$(EXESUFFIX)
	-$(FC) $(LDFLAGS) pa00.$(OBJX) check.$(OBJX) $(LIBS) -o a.$(EXESUFFIX)
build: pa00
run: ;
