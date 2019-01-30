#ifndef _OS_BASE_H
#define _OS_BASE_H

#define OS_CONSUME __attribute__((os_consumed))
#define OS_RETURNS_RETAINED __attribute__((os_returns_retained))
#define OS_RETURNS_RETAINED_ON_ZERO __attribute__((os_returns_retained_on_zero))
#define OS_RETURNS_RETAINED_ON_NONZERO __attribute__((os_returns_retained_on_non_zero))
#define OS_RETURNS_NOT_RETAINED __attribute__((os_returns_not_retained))
#define OS_CONSUMES_THIS __attribute__((os_consumes_this))

#define OSTypeID(type)   (type::metaClass)

#define OSDynamicCast(type, inst)   \
    ((type *) OSMetaClassBase::safeMetaCast((inst), OSTypeID(type)))

#define OSTypeAlloc(type)   ((type *) ((type::metaClass)->alloc()))

using size_t = decltype(sizeof(int));

struct OSMetaClass;

struct OSMetaClassBase {
  static OSMetaClassBase *safeMetaCast(const OSMetaClassBase *inst,
                                       const OSMetaClass *meta);

  OSMetaClassBase *metaCast(const char *toMeta);

  virtual void retain() const;
  virtual void release() const;

  virtual void taggedRetain(const void * tag = nullptr) const;
  virtual void taggedRelease(const void * tag = nullptr) const;

  virtual void free();
  virtual ~OSMetaClassBase(){};
};

struct OSObject : public OSMetaClassBase {
  virtual ~OSObject(){}

  unsigned int foo() { return 42; }

  virtual OS_RETURNS_NOT_RETAINED OSObject *identity();

  static OSObject *generateObject(int);

  static OSObject *getObject();
  static OSObject *GetObject();

  static void * operator new(size_t size);

  static const OSMetaClass * const metaClass;
};

struct OSMetaClass : public OSMetaClassBase {
  virtual OSObject * alloc() const;
  virtual ~OSMetaClass(){}
};

#endif /* _OS_BASE_H */
