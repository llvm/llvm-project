int zero_init();
int badSrcGlobal = zero_init();
int readBadSrcGlobal() { return badSrcGlobal; }

